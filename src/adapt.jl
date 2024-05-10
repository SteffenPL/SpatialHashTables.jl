function adapt_structure(to, grid::HashGrid) 
    HashGrid(
        grid.cellwidth,
        grid.cellwidthinv,
        grid.gridsize,
        grid.origin,
        adapt_structure(to, grid.cellidx),
        adapt_structure(to, grid.pointidx),
        adapt_structure(to, grid.cellstarts),
        adapt_structure(to, grid.cellends), 
        grid.lininds,
        grid.backend,
        grid.nthreads)
end