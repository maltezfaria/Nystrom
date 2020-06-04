# defines a convention on what certain methods should be called, and attempt to
# get the appropriate method by finding the parent module that defined the
# object. Works reasonably well for "getters", and avoids having to import and
# extend methods from other packages. If local definitions of the methods exist,
# we rely on dispatch to chose those instead.

flist = (:getnodes, :getnormals, :getweights, :getelements, :getnode, :getweight, :getnormal, :getelement)
for f in flist
    @eval begin
        function $f end
        @generated function $f(Y,args...;kwargs...)
            mY = parentmodule(Y)
            return quote
                $(mY.$f)(Y,args...)
            end
        end
    end
end

# defaults
getnodes(nodes::Array) = nodes
getnodes(nodes::Array,args...) = getindex(nodes,args...)
