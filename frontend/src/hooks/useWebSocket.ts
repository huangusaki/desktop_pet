import { useEffect, useRef, useCallback } from 'react';
import { useChatStore } from '../stores/chatStore';
import type { WebSocketMessage, ChatMessage } from '../types/chat';

const WS_URL = 'ws://localhost:8765/ws';

export const useWebSocket = () => {
    const socketRef = useRef<WebSocket | null>(null);
    const pingIntervalRef = useRef<number | null>(null);
    const reconnectTimeoutRef = useRef<number | null>(null);
    const { addMessage, setTyping, setConnected } = useChatStore();

    // Store functions in ref to avoid triggering reconnections
    const storeRef = useRef({ addMessage, setTyping, setConnected });

    // Update storeRef when store functions change
    useEffect(() => {
        storeRef.current = { addMessage, setTyping, setConnected };
    }, [addMessage, setTyping, setConnected]);

    const connect = useCallback(() => {
        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                console.log('WebSocket connected');
                storeRef.current.setConnected(true);

                // Send ping every 30 seconds
                pingIntervalRef.current = setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            type: 'ping',
                            timestamp: new Date().toISOString()
                        }));
                    }
                }, 30000);
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);

                    switch (message.type) {
                        case 'message':
                            if (message.data) {
                                storeRef.current.addMessage(message.data as ChatMessage);
                            }
                            break;

                        case 'typing':
                            storeRef.current.setTyping(message.is_typing ?? false);
                            break;

                        case 'error':
                            console.error('WebSocket error:', message.message);
                            break;

                        case 'agent_mode_changed':
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
                storeRef.current.setConnected(false);
            };

            ws.onclose = () => {
                console.log('WebSocket disconnected');
                storeRef.current.setConnected(false);

                // Clear ping interval
                if (pingIntervalRef.current) {
                    clearInterval(pingIntervalRef.current);
                    pingIntervalRef.current = null;
                }

                // Reconnect after 3 seconds
                reconnectTimeoutRef.current = setTimeout(() => {
                    console.log('Attempting to reconnect...');
                    connect();
                }, 3000);
            };

            socketRef.current = ws;
        } catch (error) {
            console.error('Error creating WebSocket:', error);
            storeRef.current.setConnected(false);
        }
    }, []); // Empty dependency array - connect function is stable

    useEffect(() => {
        connect();

        return () => {
            // Cleanup on unmount
            if (pingIntervalRef.current) {
                clearInterval(pingIntervalRef.current);
            }
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (socketRef.current) {
                socketRef.current.close();
            }
        };
    }, [connect]);

    return {
        isConnected: socketRef.current?.readyState === WebSocket.OPEN
    };
};
