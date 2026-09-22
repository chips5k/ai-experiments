import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description="ToyLLM")

parser.add_argument(
    "--training_file",
    default="./data/training.txt",
    help="Path to the input file (e.g., data/training.txt)",
)

parser.add_argument(
    "--log_level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Set the logging level, defaults to ERROR",
)

parser.add_argument(
    "--train",
    action="store_true",
    help="enable training before running",
)

parser.add_argument(
    "--steps",
    default=1000,
    type=int,
    help="total training steps to run",
)

parser.add_argument(
    "--block",
    default=128,
    type=int,
    help="Block size for each batch of training, e.g number of characters in string, constrained by device",
)

parser.add_argument(
    "--batch",
    default=1024,
    type=int,
    help="Number of batches for a single training step, constrained by device",
)

local_tz = datetime.now().astimezone().tzinfo
timestamp = datetime.now(tz=local_tz).strftime("%Y%m%d_%H%M%S")
parser.add_argument(
    "--model_file",
    default=f"./data/model-{timestamp}.pt",
    help="model file you want to use/or save to, defaults to timestamped file",
)

args = parser.parse_args()

C_GREY = "\x1b[38;20m"
C_DARK_GREY = "\x1b[90m"
C_GREEN = "\x1b[32;20m"
C_YELLOW = "\x1b[33;20m"
C_BOLD_YELLOW = "\x1b[33;1m"
C_RED = "\x1b[31;20m"
C_BOLD_RED = "\x1b[31;1m"
C_BLUE = "\x1b[34;20m"
C_BOLD_BLUE = "\x1b[34;1m"
C_RESET = "\x1b[0m"


class LogColourFormatter(logging.Formatter):
    L_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"

    def __init__(self):
        super().__init__()
        self.level_colours = {
            logging.DEBUG: C_DARK_GREY,
            logging.INFO: C_GREEN,
            logging.WARNING: C_YELLOW,
            logging.ERROR: C_RED,
            logging.CRITICAL: C_BOLD_RED,
        }

    def format(self, record):
        colour = self.level_colours.get(record.levelno, C_RESET)
        formatter = logging.Formatter(f"{colour}{self.L_FORMAT}{C_RESET}")
        return formatter.format(record)


log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(LogColourFormatter())

logging.basicConfig(
    level=getattr(logging, args.log_level),
    handlers=[log_handler],
)

log = logging.getLogger(__name__)


def handle_sigint(signum, frame):
    log.warning("Ctrl+C detected! Commencing graceful shutdown...")
    # Cleanup code here if needed
    log.info("Clean-up complete. Exiting application safely.")
    print(f"{C_DARK_GREY}Gracefully terminated..{C_RESET}")
    sys.exit(0)


signal.signal(signal.SIGINT, handle_sigint)


class SingleCharacterTokenDataset(Dataset):
    def __init__(self, text: str, block_size):
        self.text = text
        self.chars = sorted(set(text))

        self.tokensByIndex = {ch: i for i, ch in enumerate(self.chars)}
        self.tokensByValue = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def encode(self, input: str):
        return [self.tokensByIndex[c] for c in input]

    def decode(self, input):
        if isinstance(input, torch.Tensor):
            input = input.view(-1).tolist()
        return "".join([self.tokensByValue[i] for i in input])

    def vocab_size(self):
        return len(self.chars)


class MultiCharacterTokenDataset(Dataset):
    def __init__(self, text: str, block_size):
        self.text = text

        # TODO - replace this with custom chunking of blocks
        self.chars = sorted(set(text))

        self.tokensByIndex = {ch: i for i, ch in enumerate(self.chars)}
        self.tokensByValue = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    def encode(self, input: str):
        return [self.tokensByIndex[c] for c in input]

    def decode(self, input):
        if isinstance(input, torch.Tensor):
            input = input.view(-1).tolist()
        return "".join([self.tokensByValue[i] for i in input])

    def vocab_size(self):
        return len(self.chars)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Simple implementation where every char has a list assigned for every possible vocab item
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next.item()


def main():
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # text block size e.g number of characters
    block_size = args.block
    batch_size = args.batch
    max_steps = args.steps
    learning_rate = 1e-2

    log.info(
        f"device={device}, block_size={block_size}, batch_size={batch_size}, max_steps={max_steps}, learning_rate={learning_rate}"
    )

    try:
        with open(args.training_file, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        log.critical(f"training_file={args.training_file} does not exist")
        print(f"training_file={args.training_file} does not exist")
        sys.exit(1)
    except Exception as e:
        log.exception("An unexpected error occurred while reading the dataset")
        print(f"An unexpected error occurred while reading the dataset: {e}")
        sys.exit(1)

    if not text.strip():
        log.critical(f"Input file {args.training_file} is empty")
        print(f"Input file {args.training_file} is empty")
        sys.exit(1)

    log.info(f"training_file={args.training_file} loaded")

    dataset = SingleCharacterTokenDataset(text, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = BigramLanguageModel(dataset.vocab_size()).to(device)

    if os.path.exists(args.model_file):
        model_file = torch.load(args.model_file, map_location=device, weights_only=True)
        model.load_state_dict(model_file["model_state"])
        model.eval()
        log.info(f"Loaded existing model_file={args.model_file}")

        if not text.strip():
            log.critical(f"model_file={args.model_file} is empty")
            print(f"model_file={args.model_file} is empty")
            sys.exit(1)

    else:
        log.info("model_file={args.model_file} empty or missing, enabling train flag")
        print(f"model_file={args.model_file} empty or missing, enabling train flag...")
        args.train = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    data_iter = iter(dataloader)

    if args.train:
        log.info("Training starting")
        model.train()
        for step in range(max_steps):
            # Grab first batch (e.g "thing" becomes {['t','h','i','n','g'],['h','i','n','g',' ']})
            try:
                input_tokens, target_token_predictions = next(data_iter)

            # reset if epoch elapsed
            except StopIteration:
                data_iter = iter(dataloader)
                input_tokens, target_token_predictions = next(data_iter)

            input_tokens, target_token_predictions = (
                input_tokens.to(device),
                target_token_predictions.to(device),
            )

            logits, loss = model(input_tokens, target_token_predictions)
            # grab the largest n(=block_size) tokens and decode them
            decoded = dataset.decode(torch.argmax(logits, dim=-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            decodedLog = "".join(decoded[:25].split())
            inputLog = "".join(dataset.decode(input_tokens)[:25].split())
            targetLog = "".join(dataset.decode(target_token_predictions)[:25].split())
            log.debug(
                f"Training step={step}, Loss={loss.item()}, Logits=`{decodedLog}`, Input=`{inputLog}`, target=`{targetLog}...`"
            )

            if step % 200 == 0:
                log.info(
                    f"Training step={step}, Loss={loss.item()}, Logits=`{decodedLog}...`, Input=`{inputLog}...`, target=`{targetLog}`"
                )

        log.info(f"Training Complete, saving changes to model_file={args.model_file}")
        os.makedirs(os.path.dirname(args.model_file), exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "vocab_size": dataset.vocab_size()},
            args.model_file,
        )
        log.info(f"saved changes to model_file={args.model_file}")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    # --- Interactive Chat Loop ---
    while True:
        print(
            f"{C_DARK_GREY}Interactive prompt mode (type 'exit' or 'quit' to stop):{C_RESET}"
        )
        user_input = input(f"{C_BOLD_YELLOW}Enter prompt:{C_RESET} {C_YELLOW}")

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        try:
            sys.stdout.write(f"{C_RESET}")
            sys.stdout.flush()
            # Encode user text to integer token IDs
            input_ids = dataset.encode(user_input)
            log.debug(f"User prompt: {user_input}")

            context = torch.tensor([input_ids], dtype=torch.long, device=device)

            # Generate 100 new tokens from the prompt
            generated_ids = model.generate(context, max_new_tokens=100)

            # Decode full output back to text (batch item 0)
            result = dataset.decode(generated_ids[0])
            log.debug(f"LLM Response=`{result}`")
            print(f"{C_BOLD_BLUE}Response:{C_RESET}{C_BLUE}")

            for token_id in model.generate_stream(context, max_new_tokens=100):
                sys.stdout.write(dataset.decode([token_id]))
                time.sleep(0.020)
                sys.stdout.flush()

            print(C_RESET)

        except KeyError as e:
            sys.stdout.write(f'{C_RED}"{e}" not supported in vocabulary.\n{C_RESET}')
            sys.stdout.flush()
            log.debug(f"Prompt contains character {e} not present in vocabulary.")

    log.info("Existing interactive prompt and shutting down")


if __name__ == "__main__":
    main()
