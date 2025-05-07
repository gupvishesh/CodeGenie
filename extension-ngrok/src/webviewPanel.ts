import * as vscode from 'vscode';

export class WebviewPanel {
  public static currentPanel: WebviewPanel | undefined;
  private static readonly viewType = 'codegeniePanel';
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];

  public static createOrShow(extensionUri: vscode.Uri) {
    const column = vscode.window.activeTextEditor
      ? vscode.window.activeTextEditor.viewColumn
      : undefined;

    // If we already have a panel, show it
    if (WebviewPanel.currentPanel) {
      WebviewPanel.currentPanel._panel.reveal(column);
      return;
    }

    // Otherwise, create a new panel
    const panel = vscode.window.createWebviewPanel(
      WebviewPanel.viewType,
      'CodeGenie',
      column || vscode.ViewColumn.One,
      {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.joinPath(extensionUri, 'media'),
          vscode.Uri.joinPath(extensionUri, 'out')
        ]
      }
    );

    WebviewPanel.currentPanel = new WebviewPanel(panel, extensionUri);
  }

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;

    // Set the webview's initial html content
    this._update();

    // Listen for when the panel is disposed
    // This happens when the user closes the panel or when the panel is closed programmatically
    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);

    // Update the content based on view changes
    this._panel.onDidChangeViewState(
      e => {
        if (this._panel.visible) {
          this._update();
        }
      },
      null,
      this._disposables
    );

    // Handle messages from the webview
    this._panel.webview.onDidReceiveMessage(
      message => {
        switch (message.command) {
          case 'alert':
            vscode.window.showErrorMessage(message.text);
            return;
        }
      },
      null,
      this._disposables
    );
  }

  public dispose() {
    WebviewPanel.currentPanel = undefined;

    // Clean up our resources
    this._panel.dispose();

    while (this._disposables.length) {
      const x = this._disposables.pop();
      if (x) {
        x.dispose();
      }
    }
  }

  private _update() {
    const webview = this._panel.webview;
    this._panel.title = "CodeGenie";
    this._panel.webview.html = this._getHtmlForWebview(webview);
  }

  private _getHtmlForWebview(webview: vscode.Webview) {
    return `<!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>CodeGenie</title>
      <style>
        body {
          padding: 20px;
          color: var(--vscode-foreground);
          font-family: var(--vscode-font-family);
          background-color: var(--vscode-editor-background);
        }
        .container {
          max-width: 800px;
          margin: 0 auto;
        }
        h1 {
          color: var(--vscode-editor-foreground);
        }
        button {
          background-color: var(--vscode-button-background);
          color: var(--vscode-button-foreground);
          border: none;
          padding: 8px 16px;
          margin: 10px 0;
          cursor: pointer;
          border-radius: 2px;
        }
        button:hover {
          background-color: var(--vscode-button-hoverBackground);
        }
        .feature {
          margin-bottom: 20px;
          padding: 15px;
          background-color: var(--vscode-editor-inactiveSelectionBackground);
          border-radius: 5px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>CodeGenie</h1>
        <p>AI-powered code assistance using DeepSeek models.</p>
        
        <div class="feature">
          <h2>Code Completion</h2>
          <p>Press <code>Ctrl+Alt+C</code> (or <code>Cmd+Alt+C</code> on Mac) to complete your code.</p>
        </div>
        
        <div class="feature">
          <h2>Ghost Suggestions</h2>
          <p>As you type, you'll see ghost suggestions. Press <code>Tab</code> to accept them.</p>
        </div>
        
        <div class="feature">
          <h2>Fill in the Middle</h2>
          <p>Press <code>Ctrl+Alt+8</code> (or <code>Cmd+Alt+8</code> on Mac) to fill gaps in your code.</p>
        </div>
        
        <div class="feature">
          <h2>Debug Code</h2>
          <p>Select code and press <code>Ctrl+Shift+8</code> (or <code>Cmd+Shift+8</code> on Mac) to debug it.</p>
        </div>
        
        <div class="feature">
          <h2>Optimize Code</h2>
          <p>Press <code>Ctrl+Shift+O</code> (or <code>Cmd+Shift+O</code> on Mac) to optimize your code.</p>
        </div>
      </div>
    </body>
    </html>`;
  }
}