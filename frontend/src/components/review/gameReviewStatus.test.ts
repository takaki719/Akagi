import { describe, expect, it } from 'vitest'
import { computeLocalReviewStatus } from './gameReviewStatus'
import type { GameRecord } from '@/types'

const record = (seat: number | null, numPlayers = 4) =>
  ({ id: 'g1', our_seat: seat, num_players: numPlayers }) as unknown as GameRecord

describe('computeLocalReviewStatus', () => {
  it('hides when the game has no recorded seat', () => {
    expect(computeLocalReviewStatus(record(null), { busy: false, path: null })).toEqual({
      kind: 'hidden',
    })
  })

  it('hides for a 3p table (review.py is 4p-only)', () => {
    expect(computeLocalReviewStatus(record(0, 3), { busy: false, path: null })).toEqual({
      kind: 'hidden',
    })
  })

  it('offers review when not yet reviewed', () => {
    expect(computeLocalReviewStatus(record(0), { busy: false, path: null })).toEqual({
      kind: 'none',
    })
  })

  it('shows reviewing while the row is busy, even with a stale path', () => {
    expect(computeLocalReviewStatus(record(0), { busy: true, path: '/x.html' })).toEqual({
      kind: 'reviewing',
    })
  })

  it('offers the cached path once generated', () => {
    expect(computeLocalReviewStatus(record(0), { busy: false, path: '/x.html' })).toEqual({
      kind: 'reviewed',
      path: '/x.html',
    })
  })
})
