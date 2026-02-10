// Video configuration
// Add your videos here; main.js will load and render them.

const videosConfig = {
    // Comparisons (Baseline vs DanceGRPO vs TeleBoost)
    // Keep T2V and I2V comparisons separate for easier maintenance.
    comparisonsT2V: [
        {
            title: "Comparison T2V Case 1",
            prompt: "Artificial flattened flowers made of paper or fabric, directly facing the camera, brightly colored, falling and spinning.",
            videos: {
                baseline: "videos/t2v/case1/baseline.mp4",
                dancegrpo: "videos/t2v/case1/dancegrpo.mp4",
                ours: "videos/t2v/case1/ours.mp4"
            }
        },
        {
            title: "Comparison T2V Case 2",
            prompt: "Brown kitten running on the asphalt road.",
            videos: {
                baseline: "videos/t2v/case2/baseline.mp4",
                dancegrpo: "videos/t2v/case2/dancegrpo.mp4",
                ours: "videos/t2v/case2/ours.mp4"
            }
        }, 
         {
            title: "Comparison T2V Case 3",
            prompt: "A muscular giant running in a strange way holding a wooden log.",
            videos: {
                baseline: "videos/t2v/case3/baseline.mp4",
                dancegrpo: "videos/t2v/case3/dancegrpo.mp4",
                ours: "videos/t2v/case3/ours.mp4"
            }
        }, 
         {
            title: "Comparison T2V Case 4",
            prompt: "Add animation in the bulbs, stars and make snow fall.",
            videos: {
                baseline: "videos/t2v/case4/baseline.mp4",
                dancegrpo: "videos/t2v/case4/dancegrpo.mp4",
                ours: "videos/t2v/case4/ours.mp4"
            }
        }, 
         {
            title: "Comparison T2V Case 5",
            prompt: "View from very far distance, big lion chasing bisen very fast",
            videos: {
                baseline: "videos/t2v/case5/baseline.mp4",
                dancegrpo: "videos/t2v/case5/dancegrpo.mp4",
                ours: "videos/t2v/case5/ours.mp4"
            }
        }, 
         {
            title: "Comparison T2V Case 6",
            prompt: "People are in horse-drawn carriage pulled by two horses.",
            videos: {
                baseline: "videos/t2v/case6/baseline.mp4",
                dancegrpo: "videos/t2v/case6/dancegrpo.mp4",
                ours: "videos/t2v/case6/ours.mp4"
            }
        }
        // Add more T2V comparison cases here...
    ],

    comparisonsI2V: [
        {
            title: "Comparison I2V Case 1",
            prompt: "The video features a close-up view of a young plant sprouting from dark, fertile soil.",
            videos: {
                baseline: "videos/i2v/case1/baseline.mp4",
                dancegrpo: "videos/i2v/case1/dancegrpo.mp4",
                ours: "videos/i2v/case1/ours.mp4"
            }
        },
        {
            title: "Comparison I2V Case 2",
            prompt: "The video showcases a bearded man in a white shirt as he engages in smoking.",
            videos: {
                baseline: "videos/i2v/case2/baseline.mp4",
                dancegrpo: "videos/i2v/case2/dancegrpo.mp4",
                ours: "videos/i2v/case2/ours.mp4"
            }
        },
        {
            title: "Comparison I2V Case 3",
            prompt: "The video depicts a city skyline viewed through a rain-covered window, emphasizing a tall communication tower as the main subject.",
            videos: {
                baseline: "videos/i2v/case3/baseline.mp4",
                dancegrpo: "videos/i2v/case3/dancegrpo.mp4",
                ours: "videos/i2v/case3/ours.mp4"
            }
        },
        {
            title: "Comparison I2V Case 4",
            prompt: "The video portrays an older man, dressed in a dark suit, showing signs of distress.",
            videos: {
                baseline: "videos/i2v/case4/baseline.mp4",
                dancegrpo: "videos/i2v/case4/dancegrpo.mp4",
                ours: "videos/i2v/case4/ours.mp4"
            }
        },
        {
            title: "Comparison I2V Case 5",
            prompt: "A serene moment where a woman and a young girl are seated on a plush.",
            videos: {
                baseline: "videos/i2v/case5/baseline.mp4",
                dancegrpo: "videos/i2v/case5/dancegrpo.mp4",
                ours: "videos/i2v/case5/ours.mp4"
            }
        },
        {
            title: "Comparison I2V Case 6",
            prompt: "A heartwarming moment where a young child, flanked by her parents, is celebrating her birthday.",
            videos: {
                baseline: "videos/i2v/case6/baseline.mp4",
                dancegrpo: "videos/i2v/case6/dancegrpo.mp4",
                ours: "videos/i2v/case6/ours.mp4"
            }
        }, 
        {
            title: "Comparison I2V Case 7",
            prompt: "The video features a woman meticulously inspecting strawberries growing in a hydroponic system on a farm.",
            videos: {
                baseline: "videos/i2v/case7/baseline.mp4",
                dancegrpo: "videos/i2v/case7/dancegrpo.mp4",
                ours: "videos/i2v/case7/ours.mp4"
            }
        },
         {
            title: "Comparison I2V Case 8",
            prompt: "The video features two workers in a workshop environment, having an animated discussion while wearing safety helmets.",  
            videos: {
                baseline: "videos/i2v/case8/baseline.mp4",
                dancegrpo: "videos/i2v/case8/dancegrpo.mp4",
                ours: "videos/i2v/case8/ours.mp4"
            }
        },
 
        // Add more I2V comparison cases here...
    ]
};

// Export (if using a module system)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = videosConfig;
}
