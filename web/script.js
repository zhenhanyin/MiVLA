// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupSmoothScrolling();
    setupAnimations();
    setupLinkHandlers();
});

// Initialize Charts
function initializeCharts() {
    // Initialize image error handling
    setupImageErrorHandling();
}

// Setup Smooth Scrolling
function setupSmoothScrolling() {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Setup Animations
function setupAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all cards and sections
    document.querySelectorAll('.method-card, .result-item, section').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        observer.observe(el);
    });
}

// Setup Link Handlers
function setupLinkHandlers() {
    // Add click handlers for paper links
    document.getElementById('paper-link')?.addEventListener('click', (e) => {
        e.preventDefault();
        showComingSoon('Paper');
    });

    document.getElementById('code-link')?.addEventListener('click', (e) => {
        e.preventDefault();
        showComingSoon('Code');
    });

    document.getElementById('video-link')?.addEventListener('click', (e) => {
        e.preventDefault();
        showComingSoon('Video');
    });

    document.getElementById('dataset-link')?.addEventListener('click', (e) => {
        e.preventDefault();
        showComingSoon('Dataset');
    });
}

// Copy Citation Function
function copyCitation() {
    const citationText = `@article{mivla2024,
    title={MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training},
    author={Author Names},
    journal={Conference/Journal Name},
    year={2024},
    url={https://example.com/mivla}
}`;

    navigator.clipboard.writeText(citationText).then(() => {
        const copyBtn = document.querySelector('.copy-btn');
        const originalText = copyBtn.innerHTML;

        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.style.background = '#28a745';

        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            copyBtn.style.background = '#667eea';
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy citation: ', err);
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = citationText;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);

        const copyBtn = document.querySelector('.copy-btn');
        const originalText = copyBtn.innerHTML;
        copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
        copyBtn.style.background = '#28a745';

        setTimeout(() => {
            copyBtn.innerHTML = originalText;
            copyBtn.style.background = '#667eea';
        }, 2000);
    });
}

// Setup Image Error Handling
function setupImageErrorHandling() {
    const images = document.querySelectorAll('img[onerror]');
    images.forEach(img => {
        // Check if image exists (for local development)
        img.addEventListener('error', function() {
            this.style.display = 'none';
            if (this.nextElementSibling && this.nextElementSibling.classList.contains('image-placeholder')) {
                this.nextElementSibling.style.display = 'block';
            }
        });
    });
}

// Show Coming Soon Modal
function showComingSoon(resourceType) {
    const modal = document.createElement('div');
    modal.className = 'coming-soon-modal';
    modal.innerHTML = `
        <div class="modal-content">
            <h3>${resourceType} Coming Soon</h3>
            <p>The ${resourceType.toLowerCase()} will be available shortly. Thank you for your patience!</p>
            <button class="modal-close-btn">Got it!</button>
        </div>
    `;

    // Add modal styles
    const style = document.createElement('style');
    style.textContent = `
        .coming-soon-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            animation: fadeIn 0.3s ease-out;
        }

        .modal-content {
            background: white;
            padding: 2.5rem;
            border-radius: 15px;
            text-align: center;
            max-width: 400px;
            animation: slideUp 0.3s ease-out;
        }

        .modal-content h3 {
            color: #2c3e50;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }

        .modal-content p {
            color: #666;
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }

        .modal-close-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.3s ease;
        }

        .modal-close-btn:hover {
            background: #5a6fd8;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
    `;

    document.head.appendChild(style);
    document.body.appendChild(modal);

    // Handle modal close
    const closeBtn = modal.querySelector('.modal-close-btn');
    const closeModal = () => {
        modal.remove();
        style.remove();
    };

    closeBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModal();
        }
    });

    // Auto-close after 3 seconds
    setTimeout(closeModal, 3000);
}

// Add some interactivity to the video overlay and placeholder
document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('demo-video');
    const videoOverlay = document.getElementById('video-overlay');
    const videoPlaceholder = document.querySelector('.video-placeholder');

    // Initialize overlay state - check if video file exists
    function checkVideoExists() {
        return fetch('videos/mivla_demo.mp4', { method: 'HEAD' })
            .then(response => response.ok)
            .catch(() => false);
    }

    // Handle video overlay click
    if (videoOverlay && video) {
        videoOverlay.addEventListener('click', () => {
            videoOverlay.style.display = 'none';
            video.play().catch(err => {
                console.log('Auto-play failed, user interaction may be required');
                videoOverlay.classList.add('show');
            });
        });

        // Add hover effect for clickable overlay
        videoOverlay.addEventListener('mouseenter', () => {
            if (videoOverlay.classList.contains('show') || videoOverlay.classList.contains('no-video')) {
                videoOverlay.style.opacity = '0.9';
            }
        });
        videoOverlay.addEventListener('mouseleave', () => {
            if (videoOverlay.classList.contains('show') || videoOverlay.classList.contains('no-video')) {
                videoOverlay.style.opacity = '1';
            }
        });
    }

    // Handle video events
    if (video) {
        // Show overlay on hover when video is paused
        video.addEventListener('mouseenter', () => {
            if (video.paused) {
                videoOverlay.classList.add('show');
            }
        });
        video.addEventListener('mouseleave', () => {
            if (video.paused) {
                videoOverlay.classList.remove('show');
            }
        });

        // Hide overlay when video plays
        video.addEventListener('play', () => {
            videoOverlay.classList.remove('show');
        });

        // Show overlay when video pauses
        video.addEventListener('pause', () => {
            videoOverlay.classList.add('show');
        });

        // Check if video sources exist and set initial state
        checkVideoExists().then(exists => {
            if (!exists) {
                // No video file available
                videoOverlay.classList.add('no-video');
                videoOverlay.addEventListener('click', () => {
                    showComingSoon('Video Demo');
                });
            } else {
                // Video file exists, show subtle overlay on hover
                video.addEventListener('loadedmetadata', () => {
                    // Don't hide overlay initially, let it show on hover
                });
            }
        });

        // Show overlay if video fails to load
        video.addEventListener('error', () => {
            console.log('Video failed to load, showing overlay');
            if (videoOverlay) {
                videoOverlay.classList.add('no-video');
                videoOverlay.addEventListener('click', () => {
                    showComingSoon('Video Demo');
                });
            }
        });
    }

    // Handle placeholder (fallback)
    if (videoPlaceholder) {
        videoPlaceholder.addEventListener('click', () => {
            showComingSoon('Video Demo');
        });

        // Add hover effect
        videoPlaceholder.style.cursor = 'pointer';
        videoPlaceholder.addEventListener('mouseenter', () => {
            videoPlaceholder.style.opacity = '0.9';
        });
        videoPlaceholder.addEventListener('mouseleave', () => {
            videoPlaceholder.style.opacity = '1';
        });
    }
});

// Add keyboard navigation
document.addEventListener('keydown', function(e) {
    // Press 'Escape' to close any modals
    if (e.key === 'Escape') {
        const modal = document.querySelector('.coming-soon-modal');
        if (modal) {
            modal.remove();
            const style = document.querySelector('style[data-modal]');
            if (style) style.remove();
        }
    }
});

// Add parallax effect to header
window.addEventListener('scroll', () => {
    const header = document.querySelector('.header');
    if (header) {
        const scrolled = window.pageYOffset;
        const parallax = scrolled * 0.5;
        header.style.transform = `translateY(${parallax}px)`;
    }
});

// Add loading animation for images
document.addEventListener('DOMContentLoaded', function() {
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.animation = 'fadeIn 0.5s ease-out';
        });
    });
});

// Auto-update year in footer
document.addEventListener('DOMContentLoaded', function() {
    const footer = document.querySelector('.footer p');
    if (footer) {
        const currentYear = new Date().getFullYear();
        footer.innerHTML = `&copy; ${currentYear} MiVLA Project. All rights reserved.`;
    }
});