module Spinglass

using DelimitedFiles
export load_spinglass, random_config, anneal
export SpinConfig, SpinglassModel

include("sa.jl")

end
