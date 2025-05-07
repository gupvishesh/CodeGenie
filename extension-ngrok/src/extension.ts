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

