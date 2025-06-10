# CS231N-Final-Project

This Repo outlines an attempt to apply GRPO to fine tune a small VLM to solve a specific vision reasoning task. Specifically, the visual reasoning [task](https://drive.google.com/drive/folders/1Y9aGhUe4b8dNlhNGMeflJIwt2pGS23jV?usp=sharing) is essentially to observe names assigned to circle objects directed by arrows. 

<img width="571" alt="Screenshot 2025-06-10 at 1 53 44 PM" src="https://github.com/user-attachments/assets/82a5f529-6d26-4bb1-99b2-50eeaef2d4b6" />

We attempted to address this problem through fine tuning SmolVLM-500M-Instruct, a 500 million parameter VLM that was built by Huggingface as a memory efficient yet powerful model for its size. We first tried it on a baseline (no fine tuning), a supervised tuning approach, and finally using GRPOTraining.
