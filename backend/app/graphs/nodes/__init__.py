"""LangGraph node functions for the email workflow.

A *node* is a plain (async) function that LangGraph calls with the current state and returns a
partial state update: a dict with only the keys it wants to change. LangGraph merges the update
into the state (using reducers for annotated keys) and checkpoints it. Nodes never mutate the
state they receive. The graphs that wire these nodes together live in `app/graphs/`.
"""
