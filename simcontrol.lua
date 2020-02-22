function sysCall_threadmain()
    -- Put some initialization code here


    -- Put your main loop here, e.g.:
    --
    -- while sim.getSimulationState()~=sim.simulation_advancing_abouttostop do
    --     local p=sim.getObjectPosition(objHandle,-1)
    --     p[1]=p[1]+0.001
    --     sim.setObjectPosition(objHandle,-1,p)
    --     sim.switchThread() -- resume in next simulation step
    -- end
    
    -- Constantly move the robot towards the UR5_position_goal_target
    -- UR5_target = sim.getObjectHandle('UR5_target')
    -- UR5_position_goal_target = sim.getObjectHandle('UR5_position_goal_target')
    -- while true do
    --     sim.moveToObject(UR5_target, UR5_position_goal_target,3, nil,3, 5)
    -- end
end


importShape = function(inInts, inFloats, inStrings, inBuffer)
    inMeshPath = inStrings[1]
    inShapeName = inStrings[2]
    inShapePosition = {inFloats[1], inFloats[2], inFloats[3]}
    inShapeOrientation = {inFloats[4], inFloats[5], inFloats[6]}
    inShapeShapeColor = {inFloats[7], inFloats[8], inFloats[9]}
    robotHandle = sim.getObjectHandle('UR5')
    shapeHandle = sim.importShape(0, inMeshPath, 0, 0, 1)
    sim.setObjectName(shapeHandle, inShapeName)
    sim.setObjectPosition(shapeHandle, robotHandle, inShapePosition)
    sim.setObjectOrientation(shapeHandle, robotHandle, inShapeOrientation)

    sim.setShapeColor(shapeHandle, nil, sim.colorcomponent_ambient_diffuse, inShapeShapeColor)

    sim.setObjectInt32Parameter(shapeHandle, sim.shapeintparam_static, 0)
    sim.setObjectInt32Parameter(shapeHandle, sim.shapeintparam_respondable, 1)

    -- cupHandle = sim.getObjectHandle('Cup')

    -- print('hello world!')
    -- mass, inertia, com = sim.getShapeMassAndInertia(cupHandle)
    -- mass, inertia, com = sim.getShapeMassAndInertia(shapeHandle)
    -- sim.setShapeMassAndInertia(shapeHandle, 0.5, inertia, com, nil)
    -- sim.setShapeMassAndInertia(shapeHandle, 100, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}, nil)
    sim.resetDynamicObject(shapeHandle)
    -- sim.setModelProperty(shapeHandle, sim.modelproperty_not_dynamic)
    -- print(sim.getModelProperty(shapeHandle, sim.modelproperty_not_dynamic))
    return {shapeHandle}, {}, {}, ''
end

moveObjectToPose = function(inInts, inFloats, inStrings, inBuffer)
    -- inMeshPath = inStrings[1]
    -- inShapeName = inStrings[2]
    inShapePosition = {inFloats[1], inFloats[2], inFloats[3]}
    inShapeOrientation = {inFloats[4], inFloats[5], inFloats[6]}
    -- inShapeShapeColor = {inFloats[7], inFloats[8], inFloats[9]}
    -- robotHandle = sim.getObjectHandle('UR5') -- base frame object, i.e. 'UR5'
    --robotHandle = inInts[1] -- base frame object, i.e. 'UR5'
    -- UR5_target = sim.getObjectHandle(inInts[2]) -- object to move
    objectToTeleport = inInts[1]
    objectToMoveSmoothly = inInts[2]
    baseFrameHandle = inInts[3]
    move_mode = inInts[4]
    -- UR5_position_goal_target = sim.getObjectHandle(inInts[3]) -- place to go, i.e. 'UR5_position_goal_target'
    sim.setObjectPosition(objectToTeleport, baseFrameHandle, inShapePosition)
    sim.setObjectOrientation(objectToTeleport, baseFrameHandle, inShapeOrientation)
    -- sim.moveToObject(UR5_target, UR5_position_goal_target,3, nil,3, 5)
    sim.moveToObject(objectToMoveSmoothly, objectToTeleport, move_mode, nil, inFloats[7], inFloats[8])
    -- sim.setModelProperty(shapeHandle, sim.modelproperty_not_dynamic)
    -- print(sim.getModelProperty(shapeHandle, sim.modelproperty_not_dynamic))
    return {UR5_position_goal_target}, {}, {}, ''
end


function sysCall_cleanup()
    -- Put some clean-up code here
end

-- See the user manual or the available code snippets for additional callback functions and details
