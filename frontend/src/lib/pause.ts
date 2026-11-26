/** What the owner should know (and may need to do) before retrying, by error code. */
export const PAUSE_EXPLANATIONS: Record<string, string> = {
  budget: "The daily model budget is used up. Retry after it resets (midnight UTC) or after raising it in Settings.",
  step_budget: 'The assistant used its step limit for this email. Retry to give it more steps.',
  outbound_disabled: 'Sending is switched off. Turn it on in Settings, then Retry: every check runs again before sending.',
  preflight_unavailable: 'The mail or calendar provider did not answer. Retry when it is back.',
  llm_transient: 'The model was unavailable. Retry to continue from the step that failed.',
  internal_error: 'An unexpected error stopped this run. Retry to continue from the last completed step.',
}
