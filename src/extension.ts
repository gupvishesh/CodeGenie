import * as vscode from 'vscode';
import fetch from 'cross-fetch';
import * as fs from 'fs';
import * as path from 'path';
import { WebviewPanel } from './webviewPanel';

interface ApiResponse {
    completion?: string;
    error?: string;
    response?: string;
}

interface ConnectionInfo {
    status: string;
    server_url: string;
    model_status: string;
    device: string;
}

interface ServerConfig {
    server_url: string;
}

// Debounce function to limit API calls
function debounce<F extends (...args: any[]) => any>(
    func: F,
    waitFor: number
): (...args: Parameters<F>) => Promise<ReturnType<F>> {
    let timeout: NodeJS.Timeout;
    return (...args: Parameters<F>): Promise<ReturnType<F>> =>
        new Promise(resolve => {
            if (timeout) {
                clearTimeout(timeout);
            }
            timeout = setTimeout(() => resolve(func(...args)), waitFor);
        });
}

export function activate(context: vscode.ExtensionContext) {
    console.log('CodeGenie is now active!');

    // Configuration keys
    const CONFIG_SERVER_URL = 'codegenie.serverUrl';
    const CONFIG_AUTOCONNECT = 'codegenie.autoConnect';
    
    // Create status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right);
    statusBarItem.text = "$(cloud) CodeGenie";
    statusBarItem.tooltip = "Click to configure CodeGenie server connection";
    statusBarItem.command = 'codegenie.configureServer';
    context.subscriptions.push(statusBarItem);
    statusBarItem.show();
    
    let isProcessing = false;
    let isConnected = false;

    // Get server URL from configuration
    let serverUrl = vscode.workspace.getConfiguration().get(CONFIG_SERVER_URL) as string || '';
    const autoConnect = vscode.workspace.getConfiguration().get(CONFIG_AUTOCONNECT) as boolean || false;

    // Register command to configure server
    context.subscriptions.push(
        vscode.commands.registerCommand('codegenie.configureServer', async () => {
            await promptForServerUrl();
        })
    );

    // Try to connect automatically on startup if configured
    if (autoConnect && serverUrl) {
        testConnection(serverUrl);
    } else if (!serverUrl) {
        updateStatusBar("$(cloud) CodeGenie (Not Connected)");
    }

    async function promptForServerUrl() {
        const url = await vscode.window.showInputBox({
            value: serverUrl,
            placeHolder: 'https://your-ngrok-url.ngrok.io',
            prompt: "Enter the CodeGenie server URL (ngrok URL)",
            ignoreFocusOut: true
        });
        
        if (url) {
            await vscode.workspace.getConfiguration().update(CONFIG_SERVER_URL, url, vscode.ConfigurationTarget.Global);
            serverUrl = url;
            await testConnection(serverUrl);
        }
    }

    async function testConnection(url: string) {
        try {
            updateStatusBar("$(sync~spin) Testing connection...");
            
            const response = await fetch(`${url}/connection_info`, {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                throw new Error(`Server returned ${response.status}: ${response.statusText}`);
            }
            
            const info: ConnectionInfo = await response.json();
            
            if (info.status === 'connected') {
                isConnected = true;
                updateStatusBar(`$(cloud) CodeGenie (Connected: ${info.model_status})`);
                vscode.window.showInformationMessage(`Connected to CodeGenie server at ${url}`);
            } else {
                isConnected = false;
                updateStatusBar("$(cloud-offline) CodeGenie (Error)");
                vscode.window.showErrorMessage(`Connection error: ${info.status}`);
            }
        } catch (error) {
            isConnected = false;
            updateStatusBar("$(cloud-offline) CodeGenie (Not Connected)");
            vscode.window.showErrorMessage(`Cannot connect to CodeGenie server: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    // Command to open the Webview Panel
    let startDisposable = vscode.commands.registerCommand('codegenie.start', () => {
        WebviewPanel.createOrShow(context.extensionUri);
    });

    // Command to get code completion
    let completeDisposable = vscode.commands.registerCommand('codegenie.complete', () => {
        console.log("Command codegenie.complete executed.");
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            handleCompletion(editor);
        } else {
            console.log("No active text editor.");
        }
    });

    // Command to optimize code
    let optimizeDisposable = vscode.commands.registerCommand('codegenie.optimize', async () => {
        if (!ensureConnected()) return;
        
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor found');
            return;
        }
    
        try {
            // Show progress notification
            await vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Optimizing your code...",
                cancellable: false
            }, async (progress) => {
                progress.report({ message: "Analyzing code..." });
    
                const originalCode = editor.document.getText();
    
                // Making a POST request using fetch
                const response = await fetch(`${serverUrl}/optimize`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        text: originalCode,
                        type: 'optimize'
                    })
                });
    
                if (!response.ok) {
                    vscode.window.showErrorMessage(`Failed to optimize code: ${response.statusText}`);
                    return;
                }
    
                const data = await response.json();
                const optimizedCode = data.completion;
    
                if (!optimizedCode) {
                    vscode.window.showErrorMessage('No optimized code returned from the server');
                    return;
                }
    
                progress.report({ message: "Optimization complete" });
    
                // Show apply confirmation
                const selection = await vscode.window.showInformationMessage(
                    "Code optimization complete. Apply changes?",
                    { modal: true },
                    "Apply", "Cancel"
                );
    
                if (selection === "Apply") {
                    progress.report({ message: "Applying changes..." });
    
                    // Apply the optimized code
                    await editor.edit((editBuilder) => {
                        const fullRange = new vscode.Range(
                            editor.document.positionAt(0),
                            editor.document.positionAt(originalCode.length)
                        );
                        editBuilder.replace(fullRange, optimizedCode);
                    });
    
                    vscode.window.showInformationMessage("Optimized code applied successfully!");
                }
            });
        } catch (error) {
            if (error instanceof Error) {
                vscode.window.showErrorMessage(`Failed to optimize code: ${error.message}`);
            } else {
                vscode.window.showErrorMessage('Failed to optimize code: An unknown error occurred.');
            }
            console.error('Optimization error:', error);
        }
    });
    
    // Command for filling in the middle
    let fillMiddleDisposable = vscode.commands.registerCommand('codegenie.fillInTheMiddle', async () => {
        if (!ensureConnected()) return;
        
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage("No active editor found.");
            return;
        }
        
        const document = editor.document;
        const selection = editor.selection;
        
        // Get selected text or text from the start to the cursor
        let text = selection.isEmpty
            ? document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active))
            : document.getText(selection);
        
        // Show error if there's no text to work with
        if (!text.trim()) {
            vscode.window.showErrorMessage("No text selected or before cursor to fill.");
            return;
        }
        
        // Update status bar
        updateStatusBar("$(sync~spin) Generating middle fill...");
        
        // Get result from your custom middle-fill function
        const result = await getMiddleFill(text);
        
        // Insert result only if editor is still active and result is available
        if (result && editor === vscode.window.activeTextEditor) {
            editor.edit(editBuilder => {
                editBuilder.insert(selection.end, result);
            });
        }
        
        // Reset status bar
        updateStatusBar(isConnected ? "$(cloud) CodeGenie (Connected)" : "$(cloud-offline) CodeGenie (Not Connected)");
    });

    // Debug code command
    let debugDisposable = vscode.commands.registerCommand('codegenie.debugCode', async () => {
        if (!ensureConnected()) return;
        
        const editor = vscode.window.activeTextEditor;
        
        if (!editor) {
            vscode.window.showErrorMessage('No active editor found!');
            return;
        }

        const selection = editor.selection;
        const code = editor.document.getText(selection);

        if (!code.trim()) {
            vscode.window.showErrorMessage('No code selected for debugging!');
            return;
        }

        // Create output channel
        const outputChannel = vscode.window.createOutputChannel('DeepSeek Debugger');
        outputChannel.show();
        outputChannel.appendLine('--- Starting Debugging ---');

        try {
            // Show progress notification
            vscode.window.withProgress({
                location: vscode.ProgressLocation.Notification,
                title: "Debugging code...",
                cancellable: false
            }, async (progress) => {
                progress.report({ message: "Analyzing code..." });

                const response = await fetch(`${serverUrl}/debug`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        prompt: `Please debug this code:\n\`\`\`\n${code}\n\`\`\``
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                const debugOutput = result.response;
                
                // Clear and show debug output
                outputChannel.clear();
                outputChannel.appendLine('--- Debugging Results ---');
                outputChannel.appendLine(debugOutput);
                
                // Show completion message
                vscode.window.showInformationMessage('Debugging completed! Results in "DeepSeek Debugger" output panel.');
            });

        } catch (error: any) {
            outputChannel.appendLine('--- Debugging Error ---');
            
            if (error.response) {
                outputChannel.appendLine(`Status: ${error.response.status}`);
                outputChannel.appendLine(`Data: ${JSON.stringify(error.response.data, null, 2)}`);
            } else if (error.message) {
                outputChannel.appendLine(`Error: ${error.message}`);
            } else {
                outputChannel.appendLine(`Unknown error: ${error}`);
            }
            vscode.window.showErrorMessage('Debugging failed. See output panel for details.');
        }
    });

    // Add all disposables to context
    context.subscriptions.push(
        startDisposable, 
        completeDisposable, 
        fillMiddleDisposable, 
        optimizeDisposable,
        debugDisposable
    );

    // Debounced functions
    const debouncedCompletion = debounce(getCompletion, 750);
    const debouncedGhostCompletion = debounce(getGhostCompletion, 700);

    // Completion functions
    async function getCompletion(text: string): Promise<string> {
        if (!ensureConnected()) return '';
        
        try {
            console.log("Sending text to API:", text.substring(0, 100) + "...");
            updateStatusBar("$(sync~spin) Generating completion...");
            
            const response = await fetch(`${serverUrl}/complete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json() as ApiResponse;
            console.log("API response received");

            if (data.error) {
                throw new Error(data.error);
            }

            return data.completion ?? '';
        } catch (error) {
            if (error instanceof Error) {
                console.error("API error:", error.message);
                throw new Error(`API error: ${error.message}`);
            }
            console.error("Unknown API error:", error);
            throw error;
        } finally {
            updateStatusBar(isConnected ? "$(cloud) CodeGenie (Connected)" : "$(cloud-offline) CodeGenie (Not Connected)");
        }
    }

    async function handleCompletion(editor: vscode.TextEditor) {
        if (isProcessing || !ensureConnected()) return;

        try {
            isProcessing = true;
            updateStatusBar("$(sync~spin) Generating completion...");

            const document = editor.document;
            const selection = editor.selection;

            let text: string;
            if (!selection.isEmpty) {
                text = document.getText(selection);
                console.log("Selected text for completion");
            } else {
                text = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active));
                console.log("Text up to cursor for completion");
            }

            if (!text.trim()) {
                throw new Error("No text found for completion.");
            }

            const completion = await debouncedCompletion(text);

            if (editor === vscode.window.activeTextEditor && document === editor.document) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(selection.end, completion);
                    console.log("Completion inserted");
                });
            } else {
                console.log("Editor or document changed, completion not inserted.");
            }
        } catch (error) {
            if (error instanceof Error) {
                vscode.window.showErrorMessage(error.message);
                console.error("Completion error:", error.message);
            } else {
                console.error("Unknown completion error:", error);
            }
        } finally {
            isProcessing = false;
            updateStatusBar(isConnected ? "$(cloud) CodeGenie (Connected)" : "$(cloud-offline) CodeGenie (Not Connected)");
        }
    }

    // Ghost Suggestion Logic
    let isInsertingSuggestion = false;
    let currentSuggestion = '';
    const suggestionDecoration = vscode.window.createTextEditorDecorationType({
        after: {
            color: '#6A9955',
            fontStyle: 'italic',
        },
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
    });

    async function showSuggestion(editor: vscode.TextEditor, suggestion: string) {
        if (!editor) return;
        
        const position = editor.selection.active;
        const decoration = {
            range: new vscode.Range(position, position),
            renderOptions: {
                after: {
                    contentText: suggestion,
                },
            },
        };
        editor.setDecorations(suggestionDecoration, [decoration]);
    }

    function clearSuggestion(editor: vscode.TextEditor | undefined) {
        if (!editor) return;
        editor.setDecorations(suggestionDecoration, []);
        currentSuggestion = '';
    }

    async function getGhostCompletion(text: string): Promise<string> {
        if (!ensureConnected()) return '';
        
        try {
            // Show status bar for ghost completion
            updateStatusBar("$(sync~spin) Generating suggestion...");
            
            const res = await fetch(`${serverUrl}/hf-complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: text }),
            });
            
            if (!res.ok) {
                throw new Error(`HTTP error! status: ${res.status}`);
            }
            
            const data = await res.json();
            return data.completion?.trim() || '';
        } catch (e) {
            console.error('Ghost completion error:', e);
            return '';
        } finally {
            // Reset status bar
            updateStatusBar(isConnected ? "$(cloud) CodeGenie (Connected)" : "$(cloud-offline) CodeGenie (Not Connected)");
        }
    }

    // Command to accept ghost suggestion
    context.subscriptions.push(
        vscode.commands.registerCommand('codegenie.acceptGhostSuggestion', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && currentSuggestion) {
                isInsertingSuggestion = true;
                editor.edit(editBuilder => {
                    editBuilder.insert(editor.selection.active, currentSuggestion);
                }).then(() => {
                    clearSuggestion(editor);
                    isInsertingSuggestion = false;
                });
            }
        })
    );

    // Handle user typing (and ghost suggestion logic)
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async (event) => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || event.document !== editor.document || !isConnected) return;

            // Skip suggestion if just inserted
            if (isInsertingSuggestion) return;

            const position = editor.selection.active;
            const text = editor.document.getText(new vscode.Range(new vscode.Position(0, 0), position));

            const suggestion = await debouncedGhostCompletion(text);
            if (editor === vscode.window.activeTextEditor && suggestion) {
                currentSuggestion = suggestion;
                showSuggestion(editor, currentSuggestion);
            } else if (editor === vscode.window.activeTextEditor) {
                clearSuggestion(editor);
            }
        })
    );

    // Status bar management functions
    function updateStatusBar(message: string) {
        statusBarItem.text = message;
        statusBarItem.show();
    }
    
    // Helper function to check connection
    function ensureConnected(): boolean {
        if (!serverUrl || !isConnected) {
            vscode.window.showErrorMessage('CodeGenie is not connected to a server. Please configure the server connection.');
            promptForServerUrl();
            return false;
        }
        return true;
    }

    // Function to handle fill in the middle
    async function getMiddleFill(text: string): Promise<string> { 
        if (!ensureConnected()) return '';
        
        try { 
            updateStatusBar("$(sync~spin) Generating middle fill...");
            
            const response = await fetch(`${serverUrl}/fill_in_the_middle`, { 
                method: 'POST', 
                headers: { 'Content-Type': 'application/json' }, 
                body: JSON.stringify({ text }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            return data.completion ?? '';
        } catch (error) {
            if (error instanceof Error) {
                vscode.window.showErrorMessage(`Fill-in-the-middle error: ${error.message}`);
                console.error("Fill-in-the-middle error:", error.message);
            } else {
                console.error("Unknown fill-in-the-middle error:", error);
            }
            return '';
        } finally {
            updateStatusBar(isConnected ? "$(cloud) CodeGenie (Connected)" : "$(cloud-offline) CodeGenie (Not Connected)");
        }
    }
}

export function deactivate() {
    console.log('CodeGenie extension is now deactivated');
}