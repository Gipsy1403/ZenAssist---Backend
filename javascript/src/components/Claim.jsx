'use client';

import styles from './Claim.module.css';
import { useState } from 'react';

export default function ClaimComponent({ claim, isSelected, onClick, onTagClick, onAutoTag }) {
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    const handleTagClick = (e) => {
        e.stopPropagation();

        if (claim.tag && onTagClick) {
            onTagClick(claim.tag);
        }
    };

//     const handleAutoTag = async () => {
//         if (onAutoTag) {
//             await onAutoTag(claim);
//         }
//     };
     const handleAutoTag = async (e) => {
		e.stopPropagation();

		setError('');
		setIsLoading(true);

		try {
			await onAutoTag(claim);
		} catch (err) {
			setError(err.message || 'Auto-tag failed');
		} finally {
			setIsLoading(false);
		}
	};

    return (
        <div
            className={`${styles.container} ${isSelected ? styles.selected : ''}`}
            onClick={onClick}
            role="button"
            tabIndex={0}
            aria-pressed={isSelected}
        >
            <div className={styles.content}>
                <p
                    className={styles.text}
                    id={`claim-content-${claim.id}`}
                >
                    {claim.content}
                </p>
                {claim.tag && (
                    <button
                        className={styles.tag}
                        onClick={handleTagClick}
                        aria-label={`Navigate to ${claim.tag} inbox`}
                        title={`Go to ${claim.tag} inbox`}
                    >
                        {claim.tag}
                    </button>
                )}
                {/* <button
                    className={styles.autoTag}
                    // onClick={handleAutoTag}
                    aria-label="Auto-tag claim"
                    title="Auto-tag claim"
				onClick={(e) => {
					e.stopPropagation();
					onAutoTag(claim);
				}}
                >
                    Auto-tag
                </button> */}

			<button
                    className={styles.autoTag}
                    onClick={handleAutoTag}
                    disabled={isLoading}
                    aria-label="Auto-tag claim"
                    title="Auto-tag claim"
                >
				{isLoading ? (
					<span className={styles.spinner}></span>
				) : (
					'Auto-tag'
				)}
                </button>

			{error && (
                    <p className={styles.error}>
                        {error}
                    </p>
                )}
            </div>
        </div>
    );
}
