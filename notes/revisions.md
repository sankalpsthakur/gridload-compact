# Platform data conventions found during calibration (2026-10-07)

1. Clock.  The SMARD plugin builds naive datetimes from the UTC epoch and `tz_localize('Europe/Berlin')`s them, so the
   platform timestamp is 1 h (CET) / 2 h (CEST) EARLIER than the true UTC instant.  Verified exactly (mean abs diff 0.0) on
   AT/LU for winter (archive context) and on 50Hertz for summer (live round 12720 via the public series endpoint).
2. Hourly challenge = mean of the 4 quarter-hours of each platform hour, label = hour start (ground truth of def 5 has 4-decimal values).
3. Revisions.  The served context's last ~10 h (since local midnight) are first-published values; older points equal today's
   SMARD data.  Ground truth in the archive = first-published values: differs from today's data by mean/std (relative):
   50Hertz -1.1 % / 5.7 %, DE -0.2 % / 1.8 %, AT 0.4 % / 3.7 %, Amprion 0.0 % / 0.6 %, LU,Creos 0.2 % / 1.6 % (Jan-Mar 2026).
   The public dashboard series endpoint and the SMARD API both return the revised values.
4. Contexts are 1000 POINTS (not steps): a missing quarter-hour shifts the window (13 of 900 def-2 contexts, 181 of 900 def-5 contexts irregular).
5. Per-series anchors: each series' forecast starts one step after ITS last context point; in the archive most last points are 07:45Z (def 2)
   / 08:00Z (def 5) but lagging series end hours earlier.  Live rounds end around 05:30-05:45Z (SMARD lag).
6. Duplicate context files per round in the archive are byte-identical copies.
7. Archive vs live board (found 2026-10-07).  Recomputing the arena MASE from archive forecasts + archive truth + archive context reproduces the
   official per-series MASE exactly (|diff| < 1e-9) for 20 138 of 22 984 (round, model, series) pairs of definition 2 (87.8 %): every pair of
   the 70 rounds 2026-01-01 .. 2026-03-05 and 03-09 .. 03-14, and all of round 887 (280/280).  Rounds 2026-03-06..08 (30 % of pairs) and
   2026-03-15..03-31 (100 % of pairs, round ids 960+, several with ids 6787, 7117) differ: the live board now holds only 16 of the archive's 32 models for those
   rounds and different values for them, i.e. the platform re-ran or re-scored those rounds after the archive dump.  The archive (pre-registered
   forecasts + truth + context) is the reproducible source, so all windows here are scored from the archive; TEST-F results are also reported
   split into 2026-02-08..03-14 (official-reproducible) and 2026-03-15..03-31 (archive-only).
8. Several late-March archive rounds have contexts that end 1-2 days before the registration date (e.g. round 7117, registered 03-31,
   context end 03-29 00:30Z, horizon 03-29 01:00Z .. 03-30 00:45Z): per-series anchors matter and the DST change of 2026-03-29 falls inside.
