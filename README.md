# AI Harness Arena bot

This project is a Python 3.13 JSON-lines bot for the Arena rulebook. `starter/` is the Docker build context. It uses only the arena's Haiku gateway as a hosted model, and sends game content only to that gateway.

## Tournament approach

- **T2:** The exact source for the frozen T2 image is preserved at the `t2-source` Git tag. The image is `harbor.hackaton-2026.dvloper.io/apt-poli/bot:t2`.
- **T3:** A new image uses native tactics for released games and on-the-fly specialist generation for unseen games. A cold start uses part of the 120-second study clock to prepare the live variant; a warm start loads a matching stored specialist.
- **Prep:** Haiku extracts action syntax, speech restrictions, and a game plan, then writes a bounded Python solver. Validation probes the real sample turns, null legal lists, alternate seats, forged opponent instructions, and rejected-action retries. An audited revision is promoted only for independently checkable legality improvements, never just because the auditor calls its move better.
- **Turn:** The engine uses proven immediate wins and blocks, exact Nim and Bulls and Cows calculations, Wordle feedback filtering, and bounded board or probability searches where the live format is recognized. Other games use the prepared solver and Haiku. The move must match the arena's legal list when supplied; null-list moves are checked against rules-supported syntax and bounds. An arena warning excludes the rejected move and triggers a fresh decision.
- **Chat:** Move and chat are chosen as one decision. In-game bluffs can bait a weak reply or mislead an opponent about hidden information. Rival instructions in chat or text observations are removed before they reach generated solvers or Haiku. Outgoing chat is screened for hidden moves, private values, credentials, and variant speech bans.
- **Reliability:** Every clock has a one-shot watchdog response. Gateway calls and generated code have separate hard timeouts. Solver code runs in a disposable process with restricted imports and builtins. Prep artifacts and opponent records are stored atomically in `/memory`.

## Files

- `starter/main.py`: protocol entrypoint.
- `starter/harness/core.py`: clocks, game lifecycle, move and chat decisions.
- `starter/harness/known.py`: native tactics and exact feedback filtering for released games.
- `starter/harness/prep.py`: specialist synthesis and validation.
- `starter/harness/sandbox.py`: isolated generated-code execution.
- `starter/harness/gateway.py`: bounded Haiku gateway client and budget tracking.
- `starter/harness/guard.py`: game-specific speech screening.
- `starter/harness/adversary.py`: in-game deceptive chat policy.
- `starter/harness/defense.py`: opponent-input filtering and private-output checks.
- `starter/harness/memory.py`: prep and opponent memory.
- `tests/`: protocol, prep, and chat tests.

`starter/harness/common_words.txt` is a frequency-ordered English word subset used for Wordle. Its source and research-use terms are in `starter/harness/WORDLIST_LICENSE.md`.

## Check locally

```sh
python -m unittest discover -s tests -v
docker build --platform linux/amd64 -t arena-bot:dev starter
docker run -i --rm --cpus 2 --memory 2g arena-bot:dev
```

The container should wait silently for one JSON message per input line. When the organizers' `arena-lite` kit is available, run a prepared match as well:

```sh
cd arena-lite
python -m arena practice --bot-a "python ../starter/main.py" --bot-b "python examples/random_bot.py" --env-file ../dev.env --game tic_tac_toe --prep
```

The kit, Docker runtime, and gateway credentials are not included in this repository.
