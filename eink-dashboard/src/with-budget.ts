/**
 * Race a promise against a wall-clock budget. Resolves to the promise's value
 * if it settles within `budgetMs`, or `null` if the budget elapses first.
 *
 * The original promise is NOT cancelled — pass it to `ctx.waitUntil()` so any
 * still-running work (e.g. cache writes) can complete in the background.
 *
 * Rejections inside the budget propagate as-is.
 */
export async function withBudget<T>(promise: Promise<T>, budgetMs: number): Promise<T | null> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<null>((resolve) => {
    timer = setTimeout(() => resolve(null), budgetMs);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
