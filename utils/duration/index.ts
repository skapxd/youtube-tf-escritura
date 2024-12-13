import { Duration, DurationLikeObject } from 'luxon';

export const duration = (props: DurationLikeObject) => {
  return Duration.fromObject(props).as('milliseconds');
};

export const delay = (props: DurationLikeObject) =>
  new Promise((resolve) => setTimeout(resolve, duration(props)));
