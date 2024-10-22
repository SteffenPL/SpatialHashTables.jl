function adapt_structure(to, grid::BoundedGrid) 
    BoundedGrid(
        grid.cellwidth,
        grid.cellwidthinv,
        grid.gridsize,
        grid.origin,
        adapt_structure(to, grid.cellidx),
        adapt_structure(to, grid.pointidx),
        adapt_structure(to, grid.cache),
        adapt_structure(to, grid.cellstarts),
        adapt_structure(to, grid.cellends), 
        grid.strides,
        grid.backend,
        grid.nthreads,
        grid.inds)
end

function adapt_structure(to, grid::HashGrid) 
    HashGrid(
        grid.cellwidth,
        grid.cellwidthinv,
        adapt_structure(to, grid.cellidx),
        adapt_structure(to, grid.pointidx),
        adapt_structure(to, grid.cache),
        adapt_structure(to, grid.cellstarts),
        adapt_structure(to, grid.cellends), 
        grid.pseudorandom_factors,
        grid.backend,
        grid.nthreads,
        grid.inds)
end