// import * as vscode from 'vscode';
// import fetch from 'cross-fetch';
// import { WebviewPanel } from './webviewPanel';

// interface ApiResponse {
//     completion?: string;
//     error?: string;
// }

// // Debounce function to limit API calls
// function debounce<F extends (...args: any[]) => any>(
//     func: F,
//     waitFor: number
// ): (...args: Parameters<F>) => Promise<ReturnType<F>> {
//     let timeout: NodeJS.Timeout;
//     return (...args: Parameters<F>): Promise<ReturnType<F>> =>
//         new Promise(resolve => {
//             if (timeout) {
//                 clearTimeout(timeout);
//             }
//             timeout = setTimeout(() => resolve(func(...args)), waitFor);
//         });
// }

// export function activate(context: vscode.ExtensionContext) {
//     console.log('CodeGenie is now active!');

//     // Command to open the Webview Panel
//     let startDisposable = vscode.commands.registerCommand('codegenie.start', () => {
//         WebviewPanel.createOrShow(context.extensionUri);
//     });

//     // Command to get code completion
//     let completeDisposable = vscode.commands.registerCommand('codegenie.complete', () => {
//         console.log("Command codegenie.complete executed.");
//         const editor = vscode.window.activeTextEditor;
//         if (editor) {
//             handleCompletion(editor);
//         } else {
//             console.log("No active text editor.");
//         }
//     });

//     context.subscriptions.push(startDisposable, completeDisposable);

//     let isProcessing = false;
//     const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right);
//     context.subscriptions.push(statusBarItem);

//     const debouncedCompletion = debounce(getCompletion, 750);

//     async function getCompletion(text: string): Promise<string> {
//         try {
//             console.log("Sending text to API:", text);
//             const response = await fetch('http://127.0.0.1:5000/complete', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ text }),
//             });

//             if (!response.ok) {
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }

//             const data = await response.json() as ApiResponse;
//             console.log("API response:", data);

//             if (data.error) {
//                 throw new Error(data.error);
//             }

//             return data.completion ?? '';
//         } catch (error) {
//             if (error instanceof Error) {
//                 console.error("API error:", error.message);
//                 throw new Error(`API error: ${error.message}`);
//             }
//             console.error("Unknown API error:", error);
//             throw error;
//         }
//     }

//     async function handleCompletion(editor: vscode.TextEditor) {
//         if (isProcessing) return;

//         try {
//             isProcessing = true;
//             statusBarItem.text = "$(sync~spin) Generating completion...";
//             statusBarItem.show();

//             const document = editor.document;
//             const selection = editor.selection;

//             let text: string;
//             if (!selection.isEmpty) {
//                 text = document.getText(selection);
//                 console.log("Selected text:", text);
//             } else {
//                 text = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active));
//                 console.log("Text up to cursor:", text);
//             }

//             if (!text.trim()) {
//                 throw new Error("No text found for completion.");
//             }

//             const completion = await debouncedCompletion(text);

//             if (editor === vscode.window.activeTextEditor && document === editor.document) {
//                 await editor.edit(editBuilder => {
//                     editBuilder.insert(selection.end, completion);
//                     console.log("Completion inserted:", completion);
//                 });
//             } else {
//                 console.log("Editor or document changed, completion not inserted.");
//             }
//         } catch (error) {
//             if (error instanceof Error) {
//                 vscode.window.showErrorMessage(error.message);
//                 console.error("Completion error:", error.message);
//             } else {
//                 console.error("Unknown completion error:", error);
//             }
//         } finally {
//             isProcessing = false;
//             statusBarItem.hide();
//         }
//     }

//     // Handle text changes
//     const changeDocumentSubscription = vscode.workspace.onDidChangeTextDocument(
//         async (event) => {
//             const editor = vscode.window.activeTextEditor;
//             if (editor && event.document === editor.document) {
//                 const lastChange = event.contentChanges[event.contentChanges.length - 1];
//                 if (lastChange && lastChange.text.endsWith('\n')) {
//                     console.log("Text change detected, triggering completion.");
//                     handleCompletion(editor);
//                 }
//             }
//         }
//     );

//     context.subscriptions.push(changeDocumentSubscription);
// }

// export function deactivate() {}


// import * as vscode from 'vscode';
// import fetch from 'cross-fetch';
// import { WebviewPanel } from './webviewPanel';

// interface ApiResponse {
//     completion?: string;
//     error?: string;
// }

// // Debounce function to limit API calls
// function debounce<F extends (...args: any[]) => any>(
//     func: F,
//     waitFor: number
// ): (...args: Parameters<F>) => Promise<ReturnType<F>> {
//     let timeout: NodeJS.Timeout;
//     return (...args: Parameters<F>): Promise<ReturnType<F>> =>
//         new Promise(resolve => {
//             if (timeout) {
//                 clearTimeout(timeout);
//             }
//             timeout = setTimeout(() => resolve(func(...args)), waitFor);
//         });
// }

// export function activate(context: vscode.ExtensionContext) {
//     console.log('CodeGenie is now active!');

//     // Command to open the Webview Panel
//     let startDisposable = vscode.commands.registerCommand('codegenielocal.start', () => {
//         WebviewPanel.createOrShow(context.extensionUri);
//     });

//     // Command to get code completion
//     let completeDisposable = vscode.commands.registerCommand('codegenielocal.complete', () => {
//         console.log("Command codegenie.complete executed.");
//         const editor = vscode.window.activeTextEditor;
//         if (editor) {
//             handleCompletion(editor);
//         } else {
//             console.log("No active text editor.");
//         }
//     });

//     context.subscriptions.push(startDisposable, completeDisposable);

//     let isProcessing = false;
//     const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right);
//     context.subscriptions.push(statusBarItem);

//     const debouncedCompletion = debounce(getCompletion, 750);

//     async function getCompletion(text: string): Promise<string> {
//         try {
//             console.log("Sending text to API:", text);
//             const response = await fetch('http://127.0.0.1:5000/complete', {
//                 method: 'POST',
//                 headers: {
//                     'Content-Type': 'application/json',
//                 },
//                 body: JSON.stringify({ text }),
//             });

//             if (!response.ok) {
//                 throw new Error(`HTTP error! status: ${response.status}`);
//             }

//             const data = await response.json() as ApiResponse;
//             console.log("API response:", data);

//             if (data.error) {
//                 throw new Error(data.error);
//             }

//             return data.completion ?? '';
//         } catch (error) {
//             if (error instanceof Error) {
//                 console.error("API error:", error.message);
//                 throw new Error(`API error: ${error.message}`);
//             }
//             console.error("Unknown API error:", error);
//             throw error;
//         }
//     }

//     async function handleCompletion(editor: vscode.TextEditor) {
//         if (isProcessing) return;

//         try {
//             isProcessing = true;
//             statusBarItem.text = "$(sync~spin) Generating completion...";
//             statusBarItem.show();

//             const document = editor.document;
//             const selection = editor.selection;

//             let text: string;
//             if (!selection.isEmpty) {
//                 text = document.getText(selection);
//                 console.log("Selected text:", text);
//             } else {
//                 text = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active));
//                 console.log("Text up to cursor:", text);
//             }

//             if (!text.trim()) {
//                 throw new Error("No text found for completion.");
//             }

//             const completion = await debouncedCompletion(text);

//             if (editor === vscode.window.activeTextEditor && document === editor.document) {
//                 await editor.edit(editBuilder => {
//                     editBuilder.insert(selection.end, completion);
//                     console.log("Completion inserted:", completion);
//                 });
//             } else {
//                 console.log("Editor or document changed, completion not inserted.");
//             }
//         } catch (error) {
//             if (error instanceof Error) {
//                 vscode.window.showErrorMessage(error.message);
//                 console.error("Completion error:", error.message);
//             } else {
//                 console.error("Unknown completion error:", error);
//             }
//         } finally {
//             isProcessing = false;
//             statusBarItem.hide();
//         }
//     }

//     // Handle text changes
//     const changeDocumentSubscription = vscode.workspace.onDidChangeTextDocument(
//         async (event) => {
//             const editor = vscode.window.activeTextEditor;
//             if (editor && event.document === editor.document) {
//                 const lastChange = event.contentChanges[event.contentChanges.length - 1];
//                 if (lastChange && lastChange.text.endsWith('\n')) {
//                     console.log("Text change detected, triggering completion.");
//                     handleCompletion(editor);
//                 }
//             }
//         }
//     );

//     context.subscriptions.push(changeDocumentSubscription);
// }

// export function deactivate() {}




import * as vscode from 'vscode';
import fetch from 'cross-fetch';
import { WebviewPanel } from './webviewPanel';

interface ApiResponse {
    completion?: string;
    error?: string;
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

    context.subscriptions.push(startDisposable, completeDisposable);

    let isProcessing = false;
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right);
    context.subscriptions.push(statusBarItem);

    const debouncedCompletion = debounce(getCompletion, 750);

    async function getCompletion(text: string): Promise<string> {
        try {
            console.log("Sending text to API:", text);
            const response = await fetch('http://127.0.0.1:5000/complete', {
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
            console.log("API response:", data);

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
        }
    }

    async function handleCompletion(editor: vscode.TextEditor) {
        if (isProcessing) return;

        try {
            isProcessing = true;
            statusBarItem.text = "$(sync~spin) Generating completion...";
            statusBarItem.show();

            const document = editor.document;
            const selection = editor.selection;

            let text: string;
            if (!selection.isEmpty) {
                text = document.getText(selection);
                console.log("Selected text:", text);
            } else {
                text = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active));
                console.log("Text up to cursor:", text);
            }

            if (!text.trim()) {
                throw new Error("No text found for completion.");
            }

            const completion = await debouncedCompletion(text);

            if (editor === vscode.window.activeTextEditor && document === editor.document) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(selection.end, completion);
                    console.log("Completion inserted:", completion);
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
            statusBarItem.hide();
        }
    }

    // ---- New Suggestion Logic for /hf-complete ----

    let currentSuggestion = '';
    const suggestionDecoration = vscode.window.createTextEditorDecorationType({
        after: {
            color: '#6A9955',
            fontStyle: 'italic',
        },
        rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
    });

    async function showSuggestion(editor: vscode.TextEditor, suggestion: string) {
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

    function clearSuggestion(editor: vscode.TextEditor) {
        editor.setDecorations(suggestionDecoration, []);
        currentSuggestion = '';
    }

    const getGhostCompletion = debounce(async (text: string): Promise<string> => {
        try {
            const res = await fetch('http://127.0.0.1:5000/hf-complete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: text }),
            });
            const data = await res.json();
            return data.completion?.trim() || '';
        } catch (e) {
            console.error('Ghost completion error:', e);
            return '';
        }
    }, 700);

    context.subscriptions.push(
        vscode.commands.registerCommand('codegenie.acceptGhostSuggestion', () => {
            const editor = vscode.window.activeTextEditor;
            if (editor && currentSuggestion) {
                editor.edit(editBuilder => {
                    editBuilder.insert(editor.selection.active, currentSuggestion);
                });
                clearSuggestion(editor);
            }
        })
    );

    // Handle user typing (and ghost suggestion logic)
    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async (event) => {
            const editor = vscode.window.activeTextEditor;
            if (!editor || event.document !== editor.document) return;

            const position = editor.selection.active;
            const text = editor.document.getText(new vscode.Range(new vscode.Position(0, 0), position));

            const suggestion = await getGhostCompletion(text);
            if (editor === vscode.window.activeTextEditor && suggestion) {
                currentSuggestion = suggestion;
                showSuggestion(editor, currentSuggestion);
            } else if (editor) {
                clearSuggestion(editor);
            }
        })
    );
}

export function deactivate() {}
