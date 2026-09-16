# %% [markdown]
# # <Lesson title: the thing they will build>
#
# **You will build:** one sentence naming the artefact.
# **Time:** ~NN minutes · **Runs on:** a laptop CPU, 8 GiB RAM · **Prerequisites:** none
#
# By the end you will be able to:
# 1. Implement ...
# 2. Measure ...
# 3. Explain why ...

# %%
# Setup: everything the lesson needs, in one cell, with versions printed.
import sys
print(sys.version.split()[0])

# %% [markdown]
# ## 1. The idea, in the smallest form that is still true
#
# Short. Then straight into something that runs.

# %%
# A cell the student RUNS to see the phenomenon before they are asked to build it.

# %% [markdown]
# ## 2. Exercise 1 — <what they implement>
#
# Fill in the function. Run the checks below it; they tell you what is wrong, not just that
# something is.

# %%
def exercise_one(x):
    """One line on what it returns.

    Example:
        >>> exercise_one(2)
        4
    """
    # YOUR CODE HERE
    raise NotImplementedError


# Public checks — run these as often as you like.
def _check_one():
    assert exercise_one(2) == 4, "exercise_one(2) should be 4 — are you returning, not printing?"
    print("exercise 1 looks right")


# %% [markdown]
# ## 3. Common mistakes
#
# - The one that catches most people, and how to spot it.

# %% [markdown]
# ## 4. Self-check
#
# 1. Question probing the misconception this lesson exists to fix.
#    - (a) ... (b) ... (c) ...
#
# Answers in `solutions/`.

# %% [markdown]
# ## What you built, and where it goes next
#
# One sentence tying this to the flagship or capstone it feeds.

# %%
if __name__ == "__main__":
    _check_one()
