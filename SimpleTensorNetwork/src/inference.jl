r = 20
W = 200
vars = [
    ("A", 0.0, 0.0), ("S", 0.75, 0.0),
    ("T", 0.0, 0.3), ("L", 0.5, 0.3), ("B", 1.0, 0.3), 
    ("E", 0.25, 0.6), ("X", 0.0, 0.9), ("D", 0.75, 0.9)]
@drawsvg begin
    origin(200, 0)
    nodes = []
    for (t, x, y) in vars
        push!(nodes, node(circle, Point(x*W+0.15W, y*W+0.15W), r, :stroke))
    end
    for (k, node) in enumerate(nodes)
        LuxorGraphPlot.draw_vertex(node, stroke_color="black",
            fill_color="white", line_width=2, line_style="solid")
        LuxorGraphPlot.draw_text(node.loc, vars[k][1]; fontsize=14, color="black", fontface="")
    end
    for (i, j) in [(1, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 8), (6, 7), (6, 8)]
        LuxorGraphPlot.draw_edge(nodes[i], nodes[j], color="black", line_width=2, line_style="solid", arrow=true)
    end
end 600 W*1.3


eincode = ein"at,ex,sb,sl,tle,ebd,a,s,t,l,b,e,x,d->"

optimized_eincode = optimize_code(eincode, uniformsize(eincode, 2), TreeSA())

contraction_complexity(optimized_eincode, uniformsize(optimized_eincode, 2))

function contract(ancillas...)
    # 0 -> NO
    # 1 -> YES
    AT = [0.98 0.02; 0.95 0.05]
    EX = [0.99 0.01; 0.02 0.98]
    SB = [0.96 0.04; 0.88 0.12]
    SL = [0.99 0.01; 0.92 0.08]
    TLE = zeros(2, 2, 2)
    TLE[1,:,:] .= [1.0 0.0; 0.0 1.0]
    TLE[2,:,:] .= [0.0 1.0; 0.0 1.0]
    EBD = zeros(2, 2, 2)
    EBD[1,:,:] .= [0.8 0.2; 0.3 0.7]
    EBD[2,:,:] .= [0.2 0.8; 0.05 0.95]
    return optimized_eincode(AT, EX, SB, SL, TLE, EBD, ancillas...)[]
end


contract([0.0, 1.0], [1.0, 0.0], [1.0, 1.0], # A, S, T
        [0.0, 1.0], [1.0, 1.0], # L, B
        [1.0, 1.0], # E
        [1.0, 1.0], [1.0, 1.0] # X, D
        )

