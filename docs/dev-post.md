---
title: Which trees near me are turning this weekend? Forecasting fall color from 6,650 iNaturalist observations with TabPFN
published: false
tags: devchallenge, hf26challenge, machinelearning, python
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 1: Touch Grass](https://dev.to/challenges/hacktoberfest-week1-2026-10-05)*

**Built with PriorLabs-TabPFN.**

## What I Built

Fall color doesn't show up on a fixed date. It comes when the nights turn cold, and every fall turns cold on its own schedule. Foliage maps can tell you that a region is "near peak". They can't tell you whether the red maples on your usual walk will be red on Saturday, or whether you should wait another week.

**peakweek** answers that question for the trees around you. You type a town in the US Northeast or Mid-Atlantic, wait about a minute, and get one card for the coming weekend:

- for each of 8 common fall-color trees (red maple, sugar maple, sweetgum, northern red oak, American beech, black gum, sassafras, Norway maple), the chance that a tree of that species you come across will be **green**, **in color**, or **bare**;
- a verdict per species (**go**, **starting**, **wait**, **late**) and the day in the next two weeks with the best chance of color;
- one line on how to recognize each tree, so you know which one you're looking at once you're out there;
- one plain sentence at the top that tells you what to do.

Here is the headline for New Brunswick, New Jersey, from a forecast made on Tuesday, October 6, for the weekend of October 10–11:

> No tree here is in full color yet this weekend, but sassafras, black gums and sugar maples are starting to turn; come back around Mon, Oct 19 for northern red oaks.

And here it is for Burlington, Vermont, from the same day:

> This weekend, go see red maples, sugar maples and black gums; come back around Tue, Oct 20 for northern red oaks.

Then the screen gets out of the way. **Print card** gives you a one-page, black-and-white card to take with you. **Done, go outside** shrinks the page to that one sentence. The footer asks for one thing after your walk: add what you saw to iNaturalist with the "Leaves" annotation. Those annotations are exactly what the model learns from, so every walk makes next fall's forecast a little better.

It's for anyone who likes to go look at leaves and would rather spend that time outside than refreshing a foliage map.

## Demo

![peakweek demo: search Burlington, wait for the forecast, read the card, print it, go outside](https://raw.githubusercontent.com/Bowen1314/peakweek/main/docs/demo.gif)

*This is a real run, searching Burlington, Vermont: the server runs TabPFN on a 4-core CPU with no GPU, and the 100 seconds it spent on this forecast (the first one after the server started) are sped up 10x in the GIF. [Full video](https://github.com/Bowen1314/peakweek/blob/main/docs/demo.mp4).*

On a phone, **Use my location** near New Brunswick gives the "not yet" version of the card:

![The card on a phone near New Brunswick, NJ: no tree in full color yet, five species starting to turn](https://raw.githubusercontent.com/Bowen1314/peakweek/main/docs/screenshots/phone-3-card.png)

And this is what **Print card** produces, one black-and-white page:

![The printed card for Burlington, VT](https://raw.githubusercontent.com/Bowen1314/peakweek/main/docs/screenshots/4-print-preview.png)

There's no public deployment. Each forecast runs an open-weight model on a CPU for about a minute, and I'd rather you run it on your own laptop than wait in a queue on mine. The repo includes saved real forecasts for New Brunswick, Burlington and Pittsburgh, so you can open the full UI without installing the model:

```bash
python3 server.py --fixture examples/forecast_new_brunswick.json   # http://127.0.0.1:8770/
python3 -m peakweek --fixture examples/forecast_burlington.json      # the same card, in the terminal
```

## Code

{% embed https://github.com/Bowen1314/peakweek %}

The app itself (server, page, card, verdict logic) uses only the Python standard library. The model side needs `tabpfn` and a CPU build of PyTorch. MIT licensed. The TabPFN v2 weights are licensed separately by Prior Labs.

## How I Built It

### The data: people photographing trees

On iNaturalist, people can tag a tree photo with what its leaves are doing: *Green Leaves*, *Colored Leaves*, or *No Live Leaves*. That makes it a labeled phenology dataset that nobody had to organize. I pulled every annotated September–November observation of the 8 species in a box covering the US Northeast and Mid-Atlantic (latitude 38.5–47.5, longitude −80.5 to −66.9), from 2018 to 2025.

The API returned 10,877 observations, and most of the work was deciding what to throw away:

| dropped | observations |
|---|---:|
| planted or cultivated trees | 879 |
| "casual" quality grade | 52 |
| obscured coordinates | 244 |
| location accurate to worse than 1 km | 455 |
| conflicting annotations (every one of them "green" and "colored" on the same tree) | 856 |
| no usable leaf annotation | 1 |
| no open license (I don't redistribute these) | 1,740 |
| **kept** | **6,650** |

That leaves 3,457 green, 2,846 colored and 347 bare observations, all in `data/observations.csv` with their iNaturalist ids and license codes.

### The features: this year's weather, not just the calendar

The calendar and latitude explain a lot of fall color. The question I cared about is whether *this* fall's weather adds anything on top of that. So for every observation, the same code that the app uses (`peakweek/features.py`) computes, from Open-Meteo's ERA5 archive:

- chilling degree days since September 1 (the sum of how far each day's mean temperature fell below 20 °C);
- frost nights (≤ 0 °C) and cold nights (≤ 5 °C) since September 1;
- last week's mean minimum temperature and the last two weeks' mean temperature;
- rain over the last 30 days;
- how much warmer or colder this fall has been than the same place's other falls;

plus species, latitude, longitude, elevation, day of year and day length. Weather comes per 1-degree grid cell. A finer grid would have needed about 15,000 weighted calls to Open-Meteo, more than the free tier allows in a day. The whole project used about 7,000.

One detail I didn't expect: Open-Meteo's forecast API can also return recent past days, which looked like an easy way to get "this season so far". But for September 2026 it returned only about 50 of the 92 past days I asked for, and its rainfall ran 2.3 mm/day below the archive. A model trained on archive weather would have been reading a different instrument. So the app takes September 1 to yesterday from the same ERA5 archive as the training data, and uses the forecast API only for today and the next 14 days.

### The model: TabPFN v2, on a CPU

[TabPFN](https://github.com/PriorLabs/TabPFN) is a tabular foundation model from Prior Labs. You don't train it on your data. At prediction time you show it a set of labeled rows (the "context") along with the rows you want predicted, and it returns class probabilities with no training step at all. That suits this problem well: a few thousand rows, a mix of categorical and numeric features, and an answer where calibrated probabilities matter more than the top class.

I used the **TabPFN v2** open weights (`Prior-Labs/TabPFN-v2-clf` on Hugging Face, 29 MB, Apache 2.0 plus an attribution requirement). They download without an account and run on a CPU. On a 4-core machine with no GPU, under a hard 800 MB memory limit, one forecast (8 species × 15 days = 120 rows) takes 51–58 seconds once the server is warm (the first forecast after it starts took 99 seconds in the demo recording), using about 670 MB of memory.

The interesting choice is which rows go into the context, since on a CPU, inside 800 MB, about 1,000 rows is the practical limit. I tried four strategies on a validation year:

- **One context per species** was the obvious idea, and it broke in a TabPFN-specific way. Some species have almost no "bare" rows (sweetgum has 1 in 180), and when a class is missing from the context, TabPFN gives it near-zero probability. A single bare tree of such a species then costs a fortune in log loss: 0.577 overall, against 0.436 for one shared context. Adding a little mass to the missing classes brought it down to 0.461, still worse.
- **Nearest neighbours** in place and week (a separate context for each place and week) was 24 times slower and not measurably better (−0.002 log loss, 95% CI −0.028 to +0.022).
- **One shared context** of 1,000 rows, sampled evenly across species × leaf state, did best.
- **Averaging three such contexts** (different random samples) beat a single one in 97% of paired bootstrap resamples. That's the final setup: 3 contexts × 4 estimators.

### Is it any good? One shot at a held-out year

I split by year, so the model never sees the fall it's being tested on. I chose every setting above on 2024 (training on 2018–2023), wrote the configuration to `eval/locked.json`, and then scored 2025 **once** (training on 2018–2024; 1,725 observations). Lower is better for log loss and Brier score:

| model, 2025 test | log loss | Brier | accuracy |
|---|---:|---:|---:|
| **TabPFN v2, all features** | **0.498** | **0.285** | 0.805 |
| gradient-boosted trees (sklearn), all features | 0.505 | 0.291 | 0.802 |
| TabPFN v2, calendar only (species, place, day of year) | 0.516 | 0.291 | 0.800 |
| logistic regression, all features | 0.519 | 0.289 | 0.805 |
| logistic regression, species + day of year + latitude | 0.552 | 0.307 | 0.795 |
| climatology (how often each species was colored that week in past years) | 0.581 | 0.345 | 0.752 |

What I take from it:

- **TabPFN beat the baseline I committed to in advance.** On the 2024 validation year, logistic regression on the same features was the strongest baseline, and it tied TabPFN almost exactly (0.4305 against 0.4307). On 2025, TabPFN was better by 0.021 log loss: 95% paired-bootstrap CI −0.035 to −0.007 when resampling observations, and −0.041 to −0.003 when resampling grid-cell × week clusters.
- **This fall's weather matters.** The same TabPFN given only species, place and date was 0.018 worse (CI −0.026 to −0.011). Knowing how cold the nights have been beats knowing the date.
- **It's not a clean sweep.** Gradient-boosted trees came within 0.007 of TabPFN, which is not a significant difference. Per species, TabPFN was best for red maple, sugar maple and American beech, while simpler models did better on northern red oak, black gum, sweetgum and sassafras (45–164 test observations each). Accuracy is about 0.80 for every decent model, so the gain is in the probabilities, not in the top guess.
- **Years differ more than models do.** Every model scored worse on 2025 than on 2024 (logistic regression went from 0.431 to 0.519).

I also checked the live pipeline on this fall. There were 99 annotated observations from September 26 to October 5, 2026 that passed the same quality filters (77 green, 22 colored), all made after the training data ends. Logistic regression scored 0.436 and TabPFN 0.454. That's too few observations to rank the models (CI on the difference −0.011 to +0.046), but it shows the app's probabilities are sensible on data nobody has seen. Every run, including the ones that lost, is in `eval/RESULTS.md`.

### From probabilities to "go outside"

The card never changes the model's numbers. It only summarizes them, with rules simple enough to print on the card:

| verdict | rule (weekend average) |
|---|---|
| **late** | chance of bare branches ≥ 35% |
| **go** | chance of color ≥ 50% |
| **late** | color is already fading: falling since earlier in the window while bare branches rise |
| **starting** | chance of color ≥ 25% |
| **wait** | otherwise, with the day color is most likely |

I also had to settle on wording. "About 4 in 10 sugar maples you find this weekend should show fall color" is honest, because that's what the model predicts: one tree you come across, not the share of the canopy.

The first version of the card had a problem you'd only catch by reading it: it told Burlington, Vermont, to go see sweetgum this weekend. The weather side of the forecast wasn't the issue; the model simply has no idea where trees grow. iNaturalist has no research-grade sweetgum records at all within 50 km of Burlington. So for each place, the app now makes one more keyless call to iNaturalist and counts research-grade observations of each tree within 50 km. A tree with fewer than 5 stays on the card with its numbers, but it's listed last, dimmed, marked "Rarely recorded within 50 km", and left out of the headline. In Burlington that moves sweetgum (0 records) and sassafras (3) to the bottom. In New Brunswick, every one of the eight trees has at least 541 records nearby.

### What it can't do yet

- **It's better at "not yet" than at "too late".** People rarely photograph bare trees: only 347 of 6,650 observations are bare. So the chance of bare branches stays small even late in the window, and in early October the best day is often the last day of the forecast. "Late" is rare on the card.
- **The labels come from what people chose to photograph.** A brilliant maple gets photographed more often than a dull one.
- **The data is recent.** 2024 and 2025 make up 65% of the rows, and 2018 has only 53.
- **Weather is per 1-degree cell**, so a cold hollow and a sunny ridge in the same cell get the same weather (elevation is still a feature).

## Why Does Open Innovation Matter?

For peakweek, "open" isn't a slogan. Every layer of the project depends on something open:

- **Open weights mean no gatekeeper and no meter.** TabPFN v2 is 29 MB that downloads without an account and runs on a 4-core CPU with no GPU, inside 800 MB of memory. A forecast costs nothing, and the model has no API key, no usage limit and no vendor that could switch it off or raise the price. (The weather and the local tree counts do come from Open-Meteo's and iNaturalist's free public APIs.) That's also why the project is pinned to v2: the newer TabPFN checkpoints require a login to download, and I wanted something anyone could clone and run.
- **Open data made the model possible.** The training set is 6,650 observations from people who stopped to photograph a tree and tagged its leaves. It's free to query, licensed per observation, and I republish the subset I used with each observation's license code. The weather comes from Open-Meteo under CC BY 4.0. `scripts/build_dataset.py` rebuilds it from the public APIs, and every number in this post comes from files in the repo.
- **I could look inside and fix things.** Because the model runs in my own process, I could test four ways of building its context, find the "missing class means zero probability" problem, and measure exact memory and timing under a hard cap. A hosted prediction endpoint would only have shown me the outputs.
- **Open data in, open data out.** The card ends by asking you to log what you saw on iNaturalist. A closed app would keep that observation for itself. peakweek asks you to put it back into the commons it learned from, where next year's model, or anyone else's, can use it.

## My Agent Session

I built this with Claude Code, working as a small team of agents: one built the dataset, the features and the evaluation, and ran the TabPFN backtests on a small home server; another built the card, the page and the CLI. A coordinating session wrote the specs and checked the results. Three choices shaped the build:

1. **Only open weights that run without an account**, on hardware I already have: TabPFN v2 on a small home server's CPU, under an 800 MB memory limit, with no model weights on my laptop.
2. **Freeze the settings before looking at the test year.** The configuration was written down before 2025 was scored, and the post reports that one score, including where TabPFN didn't win.
3. **Check the card, not just the metrics.** The sweetgum-in-Vermont problem above didn't show up in any metric. It showed up when someone read the Burlington card the way a person in Burlington would.

*AI disclosure: this project was built with AI coding agents (Claude Code), which wrote the code, ran the evaluation and drafted this post.*

## Prize Categories

- **Best Use of TabPFN.** TabPFN v2 (open weights, CPU) is the forecasting model: it predicts green, colored or bare for each species and day from 6,650 historical community-science observations and this season's weather, held-out year evaluation included.
