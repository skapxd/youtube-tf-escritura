import { DurationLikeObject } from 'luxon';
import { duration, delay } from '.';

describe('duration', () => {
  // Convert valid DurationLikeObject to milliseconds
  it('should convert valid DurationLikeObject to milliseconds', () => {
    const props = { hours: 1, minutes: 30 };
    const result = duration(props);
    expect(result).toBe(5400000);
  });

  // Await for a delay
  it('should await for a delay', async () => {
    const props: DurationLikeObject = { seconds: 1 };
    const start = Date.now();
    await delay(props);
    const end = Date.now();
    const result = end - start;
    expect(result).toBeGreaterThanOrEqual(duration(props));
  });
});
