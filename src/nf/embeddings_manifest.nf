/*
 * Shared reader for workflow 06's (extract_embeddings) manifest -- one TSV row per gene under
 * <embeddings_dir>/chrN/manifest/. Replaces the per-workflow copies of `loadEmbeddingsManifest`.
 *
 * A 04 results directory does not name its pooling, reference row or stored layers (see
 * extract_embeddings.nf, "Cache key"): every row carries them as columns (pooling_axis,
 * pooling_func, embedding_layers, include_reference, ...), so a directory can hold genes
 * extracted under different settings and each consumer states what it needs via `where`.
 *
 *   loadEmbeddingsManifest(dir)                                   // pretrained Borzoi rows
 *   loadEmbeddingsManifest(dir, [pooling_axes: ['channel_dim', 'seq_dim']])
 *   loadEmbeddingsManifest(dir, [model_type: 'borzoi_random', include_reference: 'true'])
 *
 * `where` keys: model_type (default 'borzoi'); pooling_axes / pooling_funcs (lists; the row's
 * comma-separated pooling_axis / pooling_func must contain every listed value); any other key
 * is an exact match on that manifest column. Returns [gene_id: row]; rows that fail the filter
 * are left out, so a gene without the required pooling reads as "not in the manifest".
 */

def loadEmbeddingsManifest(String dir, Map where = [:]) {
    def base = new File(dir)
    if (!base.exists()) error "--embeddings_dir does not exist: ${dir}"
    def modelType = where.containsKey('model_type') ? where.model_type : 'borzoi'
    def axes = (where.pooling_axes ?: []) as List
    def funcs = (where.pooling_funcs ?: []) as List
    def exact = where.findAll { k, v -> !(k in ['model_type', 'pooling_axes', 'pooling_funcs']) }
    def idx = [:]
    base.eachFileRecurse(groovy.io.FileType.FILES) { f ->
        if (!f.path.contains('/manifest/') || !f.name.endsWith('.tsv')) return
        def lines = f.readLines()
        if (lines.size() < 2) return
        def header = lines[0].split('\t')
        def vals = lines[1].split('\t', -1)
        def row = [:]
        header.eachWithIndex { h, i -> row[h] = i < vals.size() ? vals[i] : '' }
        if (row.model_type != modelType) return
        if (axes && !(row.pooling_axis ?: '').split(',').toList().containsAll(axes)) return
        if (funcs && !(row.pooling_func ?: '').split(',').toList().containsAll(funcs)) return
        if (exact.any { k, v -> row[k] != (v as String) }) return
        idx[row.gene_id] = row
    }
    return idx
}
