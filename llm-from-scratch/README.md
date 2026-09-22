# Toy-LLM

Simple project to help me learn the basics of creating an llm from scratch


## Aim

Learn how to use the python ecosystem of tools, e.g pytorch to build, train and run custom models.

## Process

Building on my prior knowledge learnt via the ai-sandbox project. Using Gemini as a interactive study assistant, with a prompt to enforce attempting and understanding much of this myself, rather than attempting to complete it for me.

1. Step 1 is a simple single token predictor, e.g given token A predict the next token - in our case this a token is a single character. Largerly useless and outputs gibberish since it ignores prior context.
2. Step 2 is updating it to use multiple characters per token, still bigram model, but likely to yield slightly more coherant results due to the vocabulary.
3. Move away from Bigram, and incorporate prior tokens when generating input - again keeping it basic, so no multi head attention/scanning yet.
4. Will update this with interim steps when i discover the,
5. Implement the full mutli-head attention approach using Q,K,V.
6. Not sure yet
7. Experiment with non text based options e.g waveforms, pixels, input controls for game characters or robots etc...


## Demos

### Step 1 - demo - single token/character training and streaming
<figure style="margin: 0;">
  <video controls width="800" preload="metadata" src="https://github.com/user-attachments/assets/65b05d6b-de78-4d2f-b6a2-412215dd19ad">
    <a href="./media/demo.mp4">Watch the demo video.</a>
  </video>
  <figcaption><em>Figure 1: Step 1 - Real-time token streaming demonstration with single token/character.</em></figcaption>
</figure>

If the embedded player is unavailable, [watch the demo video directly](./media/demo.mp4).
