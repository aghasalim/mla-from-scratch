-- Recompute results/cache.csv from the shape parameters alone.
--
-- The KV cache table in the README is arithmetic, which is exactly why it is
-- worth checking: bench/cache.py wrote it, the figures read it back, and every
-- number downstream would agree with a mistake made once. This derives all
-- twelve rows again in SQL from n_heads, d_head, d_c, d_rope and layers, and
-- lists any row where the published file disagrees.
--
--   sqlite3 -init verify/cache.sql :memory: "" < /dev/null
--
-- Cache per token per layer, in elements:
--   MHA / GQA / MQA  2 * n_kv * d_head        K and V, one vector per KV head
--   MLA              d_c + d_rope             the latent, plus the shared rope key
.bail on
.mode csv
.import results/cache.csv published
.headers off
.mode list

CREATE TABLE cfg(config TEXT, n_heads INT, d_head INT, d_c INT, d_rope INT, layers INT);
-- DeepSeek-V2 shape, and the shape this repo actually trains.
INSERT INTO cfg VALUES ('deepseek-v2-ish', 32, 128, 512, 64, 60),
                       ('this-repo',        6,  32,  48, 16,  4);
CREATE TABLE grp(g INT);
INSERT INTO grp VALUES (2), (4), (8);

CREATE TABLE computed AS
WITH v AS (
    SELECT config, 'MHA' AS variant, 2 * n_heads * d_head AS per, layers, 1 AS ord
      FROM cfg
    UNION ALL
    SELECT config, 'GQA(g=' || g || ')', 2 * (n_heads / g) * d_head, layers, 2
      FROM cfg JOIN grp ON n_heads % g = 0
    UNION ALL
    SELECT config, 'MQA', 2 * d_head, layers, 3 FROM cfg
    UNION ALL
    SELECT config, 'MLA', d_c + d_rope, layers, 4 FROM cfg
    UNION ALL
    SELECT config, 'MLA (no decoupled RoPE)', d_c, layers, 5 FROM cfg
)
SELECT v.config, v.variant, v.per AS elem, v.layers,
       (SELECT m.per FROM v m WHERE m.config = v.config AND m.variant = 'MHA') * 1.0 / v.per
           AS reduction,
       v.per * v.layers * 131072 * 2 / 1073741824.0 AS gb
  FROM v;

-- Anything the arithmetic and the file disagree about, in either direction.
SELECT 'MISMATCH ' || c.config || ' ' || c.variant
       || ' elem ' || c.elem || ' vs ' || p.elem_per_token_per_layer
       || ' reduction ' || ROUND(c.reduction, 10) || ' vs '
       || ROUND(p.reduction_vs_mha, 10)
       || ' gb ' || ROUND(c.gb, 10) || ' vs ' || ROUND(p.gb_at_128k_fp16, 10)
  FROM computed c JOIN published p
    ON p.config = c.config AND p.variant = c.variant
 WHERE c.elem <> CAST(p.elem_per_token_per_layer AS INT)
    OR c.layers <> CAST(p.layers AS INT)
    OR ROUND(c.reduction, 10) <> ROUND(p.reduction_vs_mha, 10)
    OR ROUND(c.gb, 10) <> ROUND(p.gb_at_128k_fp16, 10);

SELECT 'MISSING FROM FILE ' || config || ' ' || variant FROM computed
 WHERE NOT EXISTS (SELECT 1 FROM published p
                    WHERE p.config = computed.config AND p.variant = computed.variant);

SELECT 'EXTRA IN FILE ' || config || ' ' || variant FROM published
 WHERE NOT EXISTS (SELECT 1 FROM computed c
                    WHERE c.config = published.config AND c.variant = published.variant);

SELECT 'rows ' || (SELECT COUNT(*) FROM published)
       || ' compared ' || (SELECT COUNT(*) FROM computed c JOIN published p
                            ON p.config = c.config AND p.variant = c.variant);
