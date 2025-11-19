import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../stores/chatStore';
import type { WebSocketMessage, ChatMessage } from '../types/chat';

const WS_URL = 'ws://localhost:8765/ws';

export const useWebSocket = () => {
    const socketRef = useRef<WebSocket | null>(null);
    const { addMessage, setTyping, setConnected } = useChatStore();

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('WebSocket connected');
                setConnected(true);

                // Send ping every 30 seconds
                const pingInterval = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ping',
                            timestamp: new Date().toISOString()
                        }));
                    }
                }, 30000);

                ws.onclose = () => {
                    clearInterval(pingInterval);
                };
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);

                    switch (message.type) {
                        case 'message':
                            if (message.data) {
                                addMessage(message.data as ChatMessage);
                            }
                            break;

                        case 'typing':
                            setTyping(message.is_typing ?? false);
                            break;

                        case 'error':
                            console.error('WebSocket error:', message.message);
                            break;

                        case 'agent_mode_changed':
                            // Handle agent mode change
                            console.log('Agent mode changed:', message.enabled);
                            break;

                        case 'pong':
                            // Pong received
                            break;
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                setConnected(false);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                setConnected(false);

                // Reconnect after 3 seconds
                setTimeout(() => {
                    connect();
                }, 3000);
            };

            socketRef.current = ws;
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            setConnected(false);
        }
    }, [addMessage, setTyping, setConnected]);

    useEffect(() => {
        connect();

        return () => {
            if (socketRef.current) {
                socketRef.current.close();
            }
        };
    }, [connect]);

    return {
        isConnected: socketRef.current?.readyState === WebSocket.OPEN
    };
};
