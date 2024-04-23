using Spinglass, LuxorGraphPlot, Graphs

gadget_not = sg_gadget_not()
@info "Logic NOT gate is represented by the spinglass model:\n$gadget_not"
gs_not = ground_states(gadget_not.sg)
@info "Ground state energy and ground states of the NOT gate:\n$gs_not"
gadget_and = sg_gadget_and()
@info "Logic AND gate is represented by the spinglass model:\n$gadget_and"
gs_and = ground_states(gadget_and.sg)
@info "Ground state energy and ground states of the AND gate:\n$gs_and"
gadget_or = sg_gadget_or()
@info "Logic OR gate is represented by the spinglass model:\n$gadget_or"
gs_or = ground_states(gadget_or.sg)
@info "Ground state energy and ground states of the OR gate:\n$gs_or"

# the following code composes a multiplier of p(2 bits) x q (2 bits)
arr = Spinglass.compose_multiplier(2, 2)
@info "Multiplier of 2 bits x 2 bits is represented by the spinglass model:\n$arr"
tt = truth_table(arr)
@info "The truth table is as follows:"
display(tt)

@info "Setting the input to [0, 1, 0, 1] to compute 2 x 2"
Spinglass.set_input!(arr, [0, 1, 0, 1])  # 2 x 2 == 4
result = truth_table(arr)
@info "The output of 2 x 2 is: $result, which is binary 4"