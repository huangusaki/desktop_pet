export interface ChatMessage {
    id: string;
    sender: string;
    text: string;
    timestamp: string;
    is_user: boolean;
    emotion?: string;
}

export interface Config {
    pet_name: string;
    user_name: string;
    pet_avatar: string;
    user_avatar: string;
    agent_mode_enabled: boolean;
    available_emotions: string[];
}

export interface WebSocketMessage {
    type: 'message' | 'typing' | 'error' | 'agent_mode_changed' | 'pong';
    data?: any;
    message?: string;
    is_typing?: boolean;
    enabled?: boolean;
    timestamp?: string;
}
