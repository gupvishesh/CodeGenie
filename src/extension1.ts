import * as vscode from 'vscode';
import fetch from 'cross-fetch';

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
    console.log('CodeGenie Local is now active!');

    let isProcessing = false;
    const statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Right
    );
    context.subscriptions.push(statusBarItem);

    async function getCompletion(text: string): Promise<string> {
        try {
            console.log("Sending text to API:", text); // Log the text being sent

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
            console.log("API response:", data); // Log the API response

            if (data.error) {
                throw new Error(data.error);
            }

            return data.completion ?? '';
        } catch (error) {
            if (error instanceof Error) {
                console.error("API error:", error.message); // Log API error
                throw new Error(`API error: ${error.message}`);
            }
            console.error("Unknown API error:", error); // Log unknown error
            throw error;
        }
    }

    const debouncedCompletion = debounce(getCompletion, 750);

    async function handleCompletion(editor: vscode.TextEditor) {
        if (isProcessing) return;

        try {
            isProcessing = true;
            statusBarItem.text = "$(sync~spin) Generating completion...";
            statusBarItem.show();

            const document = editor.document;
            const selection = editor.selection;

            // Get either the selected text or text up to cursor
            let text: string;
            if (!selection.isEmpty) {
                text = document.getText(selection);
                console.log("Selected text:", text); // Log selected text
            } else {
                text = document.getText(new vscode.Range(new vscode.Position(0, 0), selection.active));
                console.log("Text up to cursor:", text); // Log text up to cursor
            }

            if (!text.trim()) {
                throw new Error("No text found for completion.");
            }

            const completion = await debouncedCompletion(text);

            // Check if editor/document still exists and is the same
            if (editor === vscode.window.activeTextEditor &&
                document === editor.document) {
                await editor.edit(editBuilder => {
                    editBuilder.insert(selection.end, completion);
                    console.log("Completion inserted:", completion); // Log the inserted completion
                });
            } else {
                console.log("Editor or document changed, completion not inserted.");// log when completion is not inserted.
            }
        } catch (error) {
            if (error instanceof Error) {
                vscode.window.showErrorMessage(error.message);
                console.error("Completion error:", error.message); // Log completion error
            }else{
                console.error("Unknown completion error:", error); // Log unknown error
            }
        } finally {
            isProcessing = false;
            statusBarItem.hide();
        }
    }

    // Register command
    let disposable = vscode.commands.registerCommand('codegenielocal.complete', () => {
        console.log("Command codegenielocal.complete executed."); // Log command execution
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            handleCompletion(editor);
        }else{
            console.log("No active text editor."); // log when no active editor.
        }
    });

    context.subscriptions.push(disposable);

    // Handle text changes
    const changeDocumentSubscription = vscode.workspace.onDidChangeTextDocument(
        async (event) => {
            const editor = vscode.window.activeTextEditor;
            if (editor && event.document === editor.document) {
                const lastChange = event.contentChanges[event.contentChanges.length - 1];
                if (lastChange && lastChange.text.endsWith('\n')) {
                    console.log("Text change detected, triggering completion."); // Log text change trigger
                    handleCompletion(editor);
                }
            }
        }
    );

    context.subscriptions.push(changeDocumentSubscription);
}

export function deactivate() { }