import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface CustomSelectProps {
    value: string | number;
    onChange: (value: string) => void;
    options: string[];
    className?: string;
}

export const CustomSelect: React.FC<CustomSelectProps> = ({ value, onChange, options, className = '' }) => {
    const [isOpen, setIsOpen] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
                setIsOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div className={`relative ${className}`} ref={containerRef}>
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                className={`
                    w-full px-4 py-2 rounded-lg text-left flex items-center justify-between
                    bg-white/80 backdrop-blur-sm border border-slate-200/80
                    text-slate-800
                    hover:bg-white hover:border-blue-400
                    focus:outline-none focus:ring-2 focus:ring-blue-500/50
                    dark:bg-white/5 dark:border-white/10 dark:text-white
                    dark:hover:bg-white/10 dark:hover:border-white/30
                    transition-all duration-200
                `}
            >
                <span className="truncate">{value}</span>
                <motion.svg
                    animate={{ rotate: isOpen ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                    className="w-5 h-5 text-slate-400 dark:text-white/50 flex-shrink-0 ml-2"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </motion.svg>
            </button>

            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, y: -10, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -10, scale: 0.95 }}
                        transition={{ duration: 0.15, ease: "easeOut" }}
                        className="
                            absolute z-50 w-full mt-2 overflow-hidden
                            bg-white/90 backdrop-blur-xl border border-slate-200/80
                            dark:bg-slate-800/90 dark:border-white/10
                            rounded-xl shadow-xl
                        "
                    >
                        <div className="max-h-60 overflow-y-auto py-1 scrollbar-thin scrollbar-thumb-slate-300 dark:scrollbar-thumb-white/20">
                            {options.map((option) => (
                                <button
                                    key={option}
                                    type="button"
                                    onClick={() => {
                                        onChange(option);
                                        setIsOpen(false);
                                    }}
                                    className={`
                                        w-full px-4 py-2.5 text-left text-sm transition-colors duration-150
                                        ${value === option
                                            ? 'bg-blue-50 text-blue-600 font-medium dark:bg-blue-500/20 dark:text-blue-300'
                                            : 'text-slate-700 hover:bg-slate-50 dark:text-slate-200 dark:hover:bg-white/5'
                                        }
                                    `}
                                >
                                    {option}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};
