import React, { useState } from 'react';

interface PasswordInputProps {
    value: string;
    onChange: (value: string) => void;
    className?: string;
}

export const PasswordInput: React.FC<PasswordInputProps> = ({ value, onChange, className }) => {
    const [showPassword, setShowPassword] = useState(false);

    return (
        <div className="relative">
            <input
                type={showPassword ? 'text' : 'password'}
                value={value || ''}
                onChange={(e) => onChange(e.target.value)}
                className={className + ' pr-12'}
            />
            <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-white/40 hover:text-white/60"
            >
                {showPassword ? '隐藏' : '显示'}
            </button>
        </div>
    );
};
