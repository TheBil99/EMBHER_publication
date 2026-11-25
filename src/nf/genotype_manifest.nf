/*
 * Shared helper for reading workflow 04's (genotype_extraction) manifest -- the single
 * source of truth for "where is this region's cached PLINK bfile".
 *
 * Mirrors src/nf/borzoi_manifest.nf's pattern exactly, for the same reason: every
 * consumer (11, 13, 14) used to independently re-derive its own genoKey string formula
 * to name its own genotypes_cache directory. Here a lookup is a plain match against the
 * manifest row's own semantic columns (region, window_size, min_maf, max_maf,
 * vcf_filter) -- there is no cache-key string formula for a consumer to get out of step
 * with, and workflow 04's own directory-naming slug (see its own genoKey) never needs
 * to leave that file.
 *
 * Only the extracted bfile is cached here -- any downstream kinship/GRM a consumer
 * builds from it (a single or MAF-binned GRM, ...) is that
 * consumer's own on-demand computation, not a manifest product: which kinship a
 * heritability method needs depends on that method's own partitioning choice, not on
 * the region alone, so caching a shared kinship would either force every consumer onto
 * the same partitioning or multiply into a cached entry per (partitioning x power)
 * combination for a build step that's already cheap.
 *
 * Manifest layout: many small per-cache-key TSV files (one row each), same rationale as
 * borzoi_manifest.nf -- concurrent or rerun workflow-14 tasks never race on a shared
 * write. Loaded here once per consumer run into an in-memory Groovy Map.
 */

def loadGenotypeManifest(String manifestDir) {
    def idx = [:]
    def dir = new File(manifestDir)
    if (!dir.exists()) return idx
    dir.eachFileRecurse(groovy.io.FileType.FILES) { f ->
        if (!f.name.endsWith('.tsv')) return
        def lines = f.readLines()
        if (lines.size() < 2) return
        def header = lines[0].split('\t')
        def vals = lines[1].split('\t', -1)
        def row = [:]
        header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i] : '' }
        idx[genotypeQueryKey(row.product_type, row.region, row.window_size, row.min_maf,
                             row.max_maf, row.vcf_filter)] = row
    }
    return idx
}

// Builds the SAME structured key `loadGenotypeManifest` indexes rows by, from either a
// manifest row's own columns or a consumer's own request parameters -- so a lookup is
// always a plain O(1) map hit. minMaf follows the same direct `as String` formatting
// borzoiQueryKey uses for its own minMaf; maxMaf/vcfFilter go through the null/empty-safe
// `s` closure since maxMaf is optional (floor-only band) and vcfFilter is usually unset.
def genotypeQueryKey(productType, region, windowSize, minMaf, maxMaf, vcfFilter) {
    def s = { v -> (v == null || v == '') ? '' : (v as String) }
    "${productType}|${region}|${windowSize as String}|${minMaf as String}|" +
        "${s(maxMaf)}|${s(vcfFilter)}"
}

// Resolves one region's PLINK bfile (bed/bim/fam + maf.txt) against an already-loaded
// manifest index, under this run's own window_size/band/vcf_filter -- returns the
// manifest row (bed_path/bim_path/fam_path/maf_path, relative to workflow 04's results
// root) or null if not cached yet. `band` is the same [min, max-or-null] Map every
// consumer already builds for its own MAF handling (see e.g. heritability_over_layers.nf's
// parseBand).
def lookupGenoBfile(Map manifestIndex, String region, Object windowSize, Map band, Object vcfFilter) {
    manifestIndex[genotypeQueryKey('region_bfile', region, windowSize, band.min, band.max, vcfFilter)]
}
