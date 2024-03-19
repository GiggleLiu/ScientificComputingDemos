using Enzyme

# Const{T}(val)
# Argument is assumed constant and not to participate in gradient calculation.

# Active{T}(val)
# Argument is scale/immutable value to differentiate w.r.t. gradient is propagated through the return value.

# Duplicated{T}(val, shadow)
# Argument is mutable and the original output val is needed.

# DuplicatedNoNeed{T}(shadow)
# Like Duplicated, except Enzyme can assume the original result isn't nessesary.

# BatchedDuplicated{T}(val, shaodows)
# Like Duplicated, but expects a Tuple of shadow values.

# BatchedDuplicatedNoNeed{T}(shaodows)
# Like DuplicatedNoNeed, but expects a Tuple of shadow values.

function enzyme_potential_field(potential::PotentialField, distance_vector::SVector)
    _, g = Enzyme.autodiff(Reverse, potential_energy, Active, Const(potential), Active(distance_vector))[1]
    return g
end