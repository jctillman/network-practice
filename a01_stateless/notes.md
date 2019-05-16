
# Components that should be separable

# Feed in data [ BN, T, DATA ]

# For each BN
    # Feed forward results
    # BBTT
    # Get


# VARIABLES
# STATIC SINGLE EQUATIONS -- DONE
# STATIC LINKED EQUATIONS
# TOTAL-LINKED-ONE-TIME --
# 


GoalApi -- Non-Reccurrent

#
#
# VERSION 1 POC
#
# 
    i = Identity([], name='input')
    
    iw = Identity([], name='fc_w1')
    ib = Identity([], name='fc_b1')

    h1 = Relu([MathAdd([MathMult([i, iw]), ib])], name='h1')

    iw2 = Identity([], name='fc_w2')
    ib2 = Identity([], name='fc_b2')

    h2 = Relu([MathAdd([MathMult([c2, iw2]), ib2])], name='h2')

    output = Probabilize(Exponential(h2))

#
#
#
# VERSION 2
#
#
#
 
    i = Named(name='input')
    
    iw = Named(name='fc_w1')
    ib = Named(name='fc_b1')
    prep = Previous('h1')
    c1 = Concat([i, prep])

    h1 = Relu([MathAdd([MathMult([c1, iw]), ib])], name='h1')

    iw2 = Named(name='fc_w2')
    ib2 = Named(name='fc_b2')
    prep2 = Previous('h2')

    c2 = Concat([h1, prep2])
    h2 = Relu([MathAdd([MathMult([c2, iw2]), ib2])], name='h2')

    output = Probabilize(Exponential(h2))

    Model = output

    Model.forward

