"""
    gmsh(ex)

If `gmsh` has been initialized, simply execute `ex`. Otherwise, initalize `gmsh`, execute `ex`, and finalize `gmsh`.

This means the global state of `gmsh` (initialized of not) does not change when calling the `@gmsh` macro.
"""
macro gmsh(ex)
    return quote
        try
            @info "try block"
            $(esc(ex))
        catch
            @info "catch block"
            gmsh.initialize()
            $(esc(ex))
            gmsh.finalize()
        end
    end
end

function gmsh_set_meshsize(h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
end

function gmsh_set_meshorder(order)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
end

function gmsh_summary()
    cmodel = gmsh
    models = gmsh.model.list()
    for model in models
        gmsh_summary(model)
    end
end

function gmsh_summary(model)
    gmsh.model.setCurrent(model)
    @printf("List of entities in model `%s`: \n", model)
    @printf("|%10s|%10s|%10s|\n","name","dimension","tag")
    ents = gmsh.model.getEntities()
    # pgroups = gmsh.model.getPhysicalGroups()
    for ent in ents
        name = gmsh.model.getEntityName(ent...)
        dim,tag = ent
        @printf("|%10s|%10d|%10d|\n", name, dim, tag)
    end
    println()
end

# function gmsh_sphere(;radius=1,center=(0,0,0))

# end
