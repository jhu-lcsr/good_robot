/**========================================================
 * Module: gen-world-view.ts
 * Created by wjwong on 9/9/15.
 =========================================================*/
/// <reference path="gen-3d-engine.ts" />
/// <reference path="../../../../../model/genstatesdb.ts" />
/// <reference path="../../../../../model/screencapdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />
angular.module('app.generate').controller('genWorldCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', 'ngDialog', 'toaster', 'APP_CONST', 'ngTableParams', 'AppUtils', '$reactive', function ($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, ngDialog, toaster, APP_CONST, ngTableParams, apputils, $reactive) {
        "use strict";
        $reactive(this).attach($scope);
        var mult = 100; //position multiplier for int random
        //subscription error for onStop;
        var subErr = function (err) { if (err)
            console.warn("err:", arguments, err); return; };
        $scope.curState = new apputils.cCurrentState();
        setTimeout(function () {
            $scope.subscribe("genstates", function () { }, {
                onReady: function (sid) { dataReady.update('genstates'); },
                onStop: subErr
            });
            $scope.subscribe("screencaps", function () { }, {
                onReady: function (sid) { dataReady.update('screencaps'); },
                onStop: subErr
            });
        }, 10);
        var dataReady = new apputils.cDataReady(2, function () {
            updateTableStateParams();
            if ($stateParams.sid) {
                $scope.showState($stateParams.sid);
            }
            else
                $scope.$apply(function () { $rootScope.dataloaded = true; });
        });
        var updateTableStateParams = function () {
            var data = GenStates.find({}, { sort: { "_id": 1 } }).fetch();
            $scope.tableStateParams = new ngTableParams({
                count: 5,
                sorting: { created: 'desc' }
            }, {
                counts: [5, 10, 20],
                paginationMaxBlocks: 8,
                paginationMinBlocks: 2,
                data: data
            });
        };
        /**
         * Check for cube overlap and increase height based on in order cube creation so updates to mycube y is correct
         * @param mycube - current cube
         * @param used - list of cubes already created in fifo order
         * @param idxdata - index associative array to get prev cube positions
         */
        var updateYCube = function (mycube, used, idxdata) {
            var myArr = [];
            used.forEach(function (c) {
                myArr.push(c);
            });
            for (var i = 0; i < myArr.length; i++) {
                var c = idxdata[myArr[i]];
                if (myengine.intersectsMeshXYZ(mycube, c, true)) {
                    //console.warn('intersect', mycube.prop.cid, mycube.position, c.prop.cid, c.position);
                    //half of the size of the cube is from base cube other half from current cube
                    mycube.position.y = c.position.y + c.prop.size / 2 + mycube.prop.size / 2;
                }
            }
        };
        /**
         * generate cube close to anchor cube if there is none then we just generate cube via field.
         * returns null or vector3 position.
         * @param size
         * @param used
         * @param idxdata
         * @returns {*}
         */
        var genCubeNear = function (size, used, idxdata) {
            if (used.length) {
                var myArr = used; //its an array
                var halfsize = size / 2;
                var halfrad = APP_CONST.fieldsize / 4; //near radius
                var anchorIdx = myArr[apputils.rndInt(0, myArr.length - 1)];
                var aPos = idxdata[anchorIdx].position;
                var fieldmin = -(APP_CONST.fieldsize / 2) + (size / 2);
                var fieldmax = (APP_CONST.fieldsize / 2) - (size / 2);
                var min = -halfrad + halfsize;
                var max = halfrad - halfsize;
                var val = APP_CONST.fieldsize;
                var it = 0;
                while (val > fieldmax || val < fieldmin) {
                    val = apputils.rndInt(min * mult, max * mult) / mult + aPos.x;
                    if (it > 50) {
                        console.warn('it > 50 posx:', val);
                    }
                    ;
                }
                var xval = val;
                val = APP_CONST.fieldsize;
                it = 0;
                while (val > fieldmax || val < fieldmin) {
                    val = apputils.rndInt(min * mult, max * mult) / mult + aPos.z;
                    if (it > 50) {
                        console.warn('it > 50 posz:', val);
                    }
                    ;
                }
                var zval = val;
                return { anchorCid: anchorIdx, position: new BABYLON.Vector3(xval, halfsize, zval) };
            }
            console.error('no existing cubes found');
            return null;
        };
        var genCubeFar = function (size, used, idxdata) {
            if (used.length) {
                var myArr = used; //its an array
                var halfsize = size / 2;
                var halfrad = APP_CONST.fieldsize / 4; //avoid radius
                var anchorIdx = myArr[apputils.rndInt(0, myArr.length - 1)];
                var aPos = idxdata[anchorIdx].position;
                var fieldmin = -(APP_CONST.fieldsize / 2) + (size / 2);
                var fieldmax = (APP_CONST.fieldsize / 2) - (size / 2);
                var min = -halfrad + halfsize;
                var max = halfrad - halfsize;
                var val = { x: APP_CONST.fieldsize, z: APP_CONST.fieldsize };
                var it = 0;
                while (val.x > fieldmax || val.x < fieldmin ||
                    val.z > fieldmax || val.z < fieldmin ||
                    (val.x > aPos.x + min && val.x < aPos.x + max
                        && val.z > aPos.z + min && val.z < aPos.z + max)) {
                    val.x = apputils.rndInt(fieldmin * mult, fieldmax * mult) / mult;
                    val.z = apputils.rndInt(fieldmin * mult, fieldmax * mult) / mult;
                    it++;
                    if (it > 50)
                        console.warn('it > 50 pos:', val);
                }
                return { anchorCid: anchorIdx, position: new BABYLON.Vector3(val.x, halfsize, val.z) };
            }
            console.error('no existing cubes found');
            return null;
        };
        /**
         * Generate stack of the anchor cube on top of the base cube
         * @param size
         * @param used
         * @param idxdata
         * @returns {*}
         */
        var genCubeStack = function (size, used, idxdata) {
            if (used.length) {
                var myArr = used; //its an array
                var aidx = apputils.rndInt(0, myArr.length - 1); //cube to move
                var anchorIdx = myArr[aidx];
                var halfsize = idxdata[anchorIdx].prop.size / 2;
                var aPos = idxdata[anchorIdx].position;
                //console.warn('genCubeStack', anchorIdx, aPos);
                return { anchorCid: anchorIdx, position: new BABYLON.Vector3(aPos.x, halfsize, aPos.z) };
            }
            console.error('no existing cubes found');
            return null;
        };
        //todo: this is not used
        var genCubeState0 = function (used, idxdata) {
            var cid = null;
            while (cid === null || _.indexOf(used, cid) > -1) {
                cid = Number(myengine.cubesid[apputils.rndInt(0, myengine.cubesid.length - 1)]);
            }
            var max = APP_CONST.fieldsize / 2 + 0.001; //give it a little wiggle room
            var min = -max;
            var data = {
                prop: {
                    size: myengine.cubesdata[cid].meta.shape.shape_params.side_length,
                    cid: cid
                },
                position: null
            };
            var isRegen = true;
            while (isRegen) {
                if (used.length) {
                    var ltype = apputils.rndInt(0, 9);
                    if (ltype < 5) {
                        //console.warn('state0 near');
                        var cubeDat = genCubeNear(data.prop.size, used, idxdata);
                        if (cubeDat)
                            data.position = cubeDat.position;
                    }
                    else {
                        //console.warn('state0 far');
                        var cubeDat = genCubeFar(data.prop.size, used, idxdata);
                        if (cubeDat)
                            data.position = cubeDat.position;
                    }
                    if (cubeDat && cubeDat.position)
                        data.position = cubeDat.position;
                    else
                        $scope.$apply(function () {
                            toaster.pop('error', 'missing position');
                        });
                }
                else {
                    var minloc = (-(APP_CONST.fieldsize / 2) + (data.prop.size / 2)) * mult;
                    var maxloc = ((APP_CONST.fieldsize / 2) - (data.prop.size / 2)) * mult;
                    data.position = new BABYLON.Vector3(apputils.rndInt(minloc, maxloc) / mult, (data.prop.size / 2), apputils.rndInt(minloc, maxloc) / mult);
                }
                if ((data.position.x - data.prop.size / 2) >= min && (data.position.x + data.prop.size / 2) <= max &&
                    (data.position.z - data.prop.size / 2) >= min && (data.position.z + data.prop.size / 2) <= max) {
                    var cubespos = [];
                    _.each(idxdata, function (i) {
                        cubespos.push(i);
                    });
                    var anchorStack = getStackCubes(data, cubespos, null, false);
                    console.warn('output', cid, anchorStack.length);
                    if (anchorStack.length < 2)
                        isRegen = false;
                }
            }
            updateYCube(data, used, idxdata);
            used.push(cid);
            idxdata[cid] = data;
            console.warn('genCubeState0', cid, data);
            return data;
        };
        /**
         * Append moves to end of the states list
         * @param params
         */
        $scope.genStateN = function (params) {
            console.warn('genStateN', params);
            //we must get the state for this params.sid
            if ($scope.curState._id) {
                var myframe = $scope.curState;
                //if(!params.cstate) //show when we use 'show state' input
                //create a munge of cube position rotate and props
                var used = [];
                var cidlist = [];
                var cubeInWorld = {};
                var cubesused = [];
                //create updated blockmeta
                var cubemeta = {};
                var maxsize = 0;
                _.each(myframe.block_meta.blocks, function (m) {
                    cubemeta[m.id] = m;
                });
                var cstate = myframe.block_states.length;
                var block_state = [];
                var orig = myframe.block_states[cstate - 1].block_state;
                for (var i = 0; i < orig.length; i++) {
                    var pos = _.extend({}, orig[i].position);
                    var rot = _.extend({}, orig[i].rotation);
                    block_state.push({ id: orig[i].id, position: pos, rotation: rot });
                }
                _.each(block_state, function (p, i) {
                    var size = cubemeta[p.id].shape.shape_params.side_length;
                    if (maxsize < size)
                        maxsize = size;
                    var val = {
                        prop: { cid: p.id, size: size },
                        position: p.position,
                        rotation: p.rotation
                    };
                    used.push(val);
                    cubeInWorld[p.id] = val;
                    cidlist.push(p.id);
                    cubesused.push(p.id);
                });
                cubesused = _.uniq(cubesused);
                var isRegen = true;
                var cubeDat, acube, cubeStack;
                while (isRegen) {
                    //let gencube choose a cube and create a position based on it
                    var ltype = apputils.rndInt(0, 19);
                    if (cidlist.length < 2) {
                        ltype = apputils.rndInt(0, 9);
                    }
                    if (ltype < 10) {
                        if (ltype < 5) {
                            cubeDat = genCubeNear(maxsize, cidlist, cubeInWorld);
                        }
                        else {
                            cubeDat = genCubeFar(maxsize, cidlist, cubeInWorld);
                        }
                    }
                    else {
                        cubeDat = genCubeStack(maxsize, cidlist, cubeInWorld);
                    }
                    //now we randomly choose a cube outside of the anchor cube id to move to the new position
                    var mycid = cubeDat.anchorCid;
                    while (mycid == cubeDat.anchorCid && block_state.length > 1) {
                        mycid = block_state[apputils.rndInt(0, block_state.length - 1)].id;
                    }
                    acube = cubeInWorld[mycid];
                    //check Y because we will move this stack
                    cubeStack = getStackCubes(acube, used, mycid, true);
                    //check stack for more than stack of 2 - meaning no stacking on top of stacks or move stacks on another
                    var anchorStack;
                    console.warn('$scope.opt.limStack', $scope.opt.limStack);
                    if ($scope.opt.limStack) {
                        if (!cubeStack.length) {
                            //don't check Y because this is the base stack where things will move to
                            //we also don't need to reference cube but by position
                            anchorStack = getStackCubes({ position: cubeDat.position, prop: { size: maxsize } }, used, null, false);
                            if (anchorStack.length < 2)
                                isRegen = false;
                            console.warn('gen itr', $scope.curState.block_states.length, mycid, cubeStack.length, cubeDat.anchorCid, anchorStack.length);
                        }
                    }
                    else
                        isRegen = false;
                }
                //remove cubes used from the world and leave world cubes in cidlist
                cidlist.splice(_.indexOf(cidlist, acube.prop.cid), 1);
                cubeStack.forEach(function (c) {
                    cidlist.splice(_.indexOf(cidlist, c.prop.cid), 1);
                });
                var basePos = { x: acube.position.x, y: acube.position.y, z: acube.position.z }; //store base Y
                acube.position = cubeDat.position;
                acube.position.y = acube.prop.size / 2; //translate it down to the ground
                /*acube.position.x = 0;
                 acube.position.z = 0;*/
                updateYCube(acube, cidlist, cubeInWorld);
                var delta = {
                    x: acube.position.x - basePos.x,
                    y: acube.position.y - basePos.y,
                    z: acube.position.z - basePos.z
                };
                cubeStack.forEach(function (c) {
                    c.position.x += delta.x;
                    c.position.y += delta.y;
                    c.position.z += delta.z;
                });
                //rebuild frame and show
                for (var i = 0; i < block_state.length; i++) {
                    block_state[i].position = cubeInWorld[block_state[i].id].position;
                }
                myengine.updateScene({ block_state: block_state }, function () {
                    if (params.itr) {
                        //this is a iterate state generation so lets save the info
                        $scope.curcnt = params.itr + 1;
                        $scope.curitr = cstate + 1;
                        params.cubesused = cubesused;
                        setTimeout(function () {
                            waitForSSAndSave(params, nextItr(params));
                        }, 400);
                    }
                    else
                        $scope.$apply(function () {
                            toaster.pop('info', 'Generated Test Move');
                        });
                });
            }
            else
                $scope.$apply(function () {
                    toaster.pop('error', 'Missing State ID');
                });
        };
        /*$scope.showInitFrame = function(state:miGen3DEngine.iCubeState[], cb:()=>void){
         $scope.resetWorld();
         console.warn('showInitFrame', state);
         setTimeout(function(){
         state.forEach(function(s){
         var c = get3DCubeById(s.prop.cid);
         c.position = new BABYLON.Vector3(s.position.x, s.position.y, s.position.z);
         c.isVisible = true;
         if(hasPhysics) c.setPhysicsState({
         impostor: BABYLON.PhysicsEngine.BoxImpostor,
         move: true,
         mass: 5, //c.boxsize,
         friction: fric,
         restitution: rest
         });
         })
         if(cb) cb();
         }, 100);
         };*/
        /*var showFrame = function (state:iBlockStates, cb?:()=>void) {
          $scope.resetWorld();
          setTimeout(function () {
            if (state.block_state) {
              state.block_state.forEach(function (frame) {
                var c = myengine.get3DCubeById(frame.id);
                c.position = new BABYLON.Vector3(frame.position['x'], frame.position['y'], frame.position['z']);
                if (frame.rotation)
                  c.rotationQuaternion = new BABYLON.Quaternion(frame.rotation['x'], frame.rotation['y'], frame.rotation['z'], frame.rotation['w']);
                c.isVisible = true;
                if (myengine.hasPhysics) c.setPhysicsState({
                  impostor: BABYLON.PhysicsEngine.BoxImpostor,
                  move: true,
                  mass: 5, //c.boxsize,
                  friction: myengine.fric,
                  restitution: myengine.rest
                });
              });
            }
            else $scope.$apply(function () {
              toaster.pop('error', 'Missing BLOCK_STATE')
            });
            if (cb) cb();
          }, 100);
        };*/
        /*var findBy = function(type:string, key:string, collection:any){
         return _.find(collection, function(a){return key === a[type]});
         };*/
        var insertGen = function (used, cb) {
            /*var str = '';
             used.forEach(function(cid){
             var c = get3DCubeById(cid);
             str += cid + ':' + c.position.x + ':' + c.position.y + ':' + c.position.z+'\n';
             });
             var sig = md5.createHash(str);
             var mygstate = findBy('sig', sig, genstates);
             if(!mygstate){*/
            if (true) {
                console.warn($scope.curState);
                //check if we loaded states or just a frame save for an existing system
                if (!$scope.curState._id && $scope.curState.block_states && $scope.curState.block_states.length
                    && $scope.curState.block_states[0].screencap) {
                    //if there is no id for current state, there are states in it and screencap then it must be a loadstates object
                    //we have to save everything in this state and save the screen caps in another value.
                    for (var i = 0; i < $scope.curState.block_states.length; i++)
                        ;
                    var saveScreen = function (idx, list, cb) {
                        if (_.isUndefined(list[idx]))
                            return cb();
                        ScreenCaps.insert({
                            data: list[idx].screencap,
                            created: (new Date).getTime(),
                            public: true
                        }, function (err, id) {
                            if (err)
                                return console.warn('screencap Err', err);
                            delete list[idx].screencap;
                            list[idx].screencapid = id;
                            saveScreen(idx + 1, list, cb);
                        });
                    };
                    saveScreen(0, $scope.curState.block_states, function (err) {
                        if (err)
                            return $scope.$apply(function () {
                                toaster.pop('error', err.reason);
                            });
                        //remove $$hashkey from scope data - http://stackoverflow.com/questions/32173465/error-key-hashkey-must-not-start-with-angularjs
                        GenStates.insert(angular.copy($scope.curState), function (err, id) {
                            if (err)
                                return console.warn('genstates Err', err);
                            $scope.curState._id = id;
                            cb(err, $scope.curState._id);
                        });
                    });
                }
                else {
                    var max = APP_CONST.fieldsize / 2 + 0.001; //give it a little wiggle room
                    var min = -max;
                    var frame = [];
                    var meta = { blocks: [] };
                    var isValid = true;
                    used.forEach(function (cid) {
                        var c = myengine.get3DCubeById(cid);
                        if (c) {
                            if ((c.position.x - c.boxsize / 2) >= min && (c.position.x + c.boxsize / 2) <= max &&
                                (c.position.z - c.boxsize / 2) >= min && (c.position.z + c.boxsize / 2) <= max) {
                                var dat = {
                                    id: cid,
                                    position: c.position.clone(),
                                    rotation: c.rotationQuaternion.clone()
                                };
                                frame.push(dat);
                                meta.blocks.push(myengine.cubesdata[cid].meta);
                            }
                            else {
                                isValid = false;
                            }
                        }
                    });
                    if (!isValid) {
                        cb('Cube(s) Out of Bounds!');
                        return false;
                    }
                    BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
                        width: myengine.canvas.width,
                        height: myengine.canvas.height
                    }, function (b64i) {
                        var b64img = LZString.compressToUTF16(b64i);
                        ScreenCaps.insert({
                            data: b64img,
                            created: (new Date).getTime(),
                            public: true
                        }, function (err, id) {
                            if (err)
                                return console.warn('screencap err:', err);
                            if (!$scope.curState.block_states)
                                $scope.curState.block_states = [];
                            $scope.curState.block_states.push({
                                block_state: frame,
                                screencapid: id,
                                created: (new Date).getTime()
                            });
                            GenStates.insert(angular.copy($scope.curState), function (err, id) {
                                if (err)
                                    return console.warn('genstates err:', err);
                                $scope.curState._id = id;
                                var attachid = createButtons('stateimg', $scope.curState.block_states.length - 1);
                                showImage(b64img, 'Move #: ' + ($scope.curState.block_states.length - 1), attachid);
                                cb(err, $scope.curState._id);
                            });
                        });
                    });
                }
            }
            else {
                cb('State already exists!');
            }
        };
        var showImage = function (b64i, text, attachID) {
            var b64img = LZString.decompressFromUTF16(b64i);
            var eleDivID = 'div' + $('div').length; // Unique ID
            var eleImgID = 'img' + $('img').length; // Unique ID
            //var eleLabelID:string = 'h4' + $('h4').length; // Unique ID
            var htmlout = '';
            if (text)
                htmlout += '<b>' + text + '</b><br>';
            htmlout += '<img id="' + eleImgID + '" style="width:' + myengine.canvas.width * 2 / 3 + 'px;height:' + myengine.canvas.height * 2 / 3 + 'px"></img>';
            // + '<label id="'+eleLabelID+'" class="mb"> '+id+'</label>';
            var attachTo = '#galleryarea';
            if (attachID)
                attachTo = '#' + attachID;
            $('<div>').attr({
                id: eleDivID
            }).addClass('col-sm-12')
                .html(htmlout).css({}).appendTo(attachTo);
            var img = document.getElementById(eleImgID); // Use the created element
            img.src = b64img;
        };
        var checkFnSS; //store steady state check
        /**
         * check for a scene steady state before saving data.
         * providing a cb will short circuit checks for startgen or startmove functions
         * @param params
         */
        var waitForSSAndSave = function (params, cb) {
            checkFnSS = setInterval(function () {
                if (myengine.isSteadyState) {
                    clearInterval(checkFnSS);
                    insertGen(params.cubesused, cb);
                }
            }, 200);
        };
        /**
         * start generation of cubes based on number of buces, iterations, and layout type
         *
         * @param ccnt
         * @param itr
         * @param cstate
         */
        $scope.startGen = function () {
            var state = [];
            var cubeidxdata = {};
            var cubesused = [];
            var myccnt = $scope.curState.block_meta.blocks.length;
            for (var i = 0; i < myccnt; i++) {
                var dat = genCubeState0(cubesused, cubeidxdata); //save used list
                state.push({ id: dat.prop.cid, position: dat.position });
            }
            if (cubesused.length != state.length)
                console.warn('done state!!', cubesused.length, state.length);
            $('#galleryarea').empty();
            myengine.createObjects($scope.curState.block_meta.blocks);
            $scope.curState.public = true;
            $scope.curState.created = (new Date).getTime();
            $scope.curState.creator = $rootScope.currentUser._id;
            myengine.updateScene({ block_state: state }, function () {
                checkFnSS = setInterval(function () {
                    if (myengine.isSteadyState) {
                        clearInterval(checkFnSS);
                        //check if all cubes are inside the bounds of the table
                        var max = APP_CONST.fieldsize / 2 + 0.001; //give it a little wiggle room
                        var min = -max;
                        var isValid = true;
                        var len = $scope.curState.block_meta.blocks.length;
                        for (var i = 0; i < len; i++) {
                            var cid = $scope.curState.block_meta.blocks[i].id;
                            var c = myengine.get3DCubeById(cid);
                            if (c) {
                                if (!((c.position.x - c.boxsize / 2) >= min && (c.position.x + c.boxsize / 2) <= max &&
                                    (c.position.z - c.boxsize / 2) >= min && (c.position.z + c.boxsize / 2) <= max)) {
                                    isValid = false; //fail time to restart the generation
                                    i = len;
                                }
                            }
                        }
                        if (!isValid)
                            $scope.startGen();
                        else
                            $scope.$apply(function () {
                                $scope.impFilename = 'system';
                                $scope.enableImpSave = true;
                            });
                    }
                }, 100);
            });
            /*
             $scope.showInitFrame(state, function(){
             var params = {cubesused: cubesused, creator: 'system'};
             //we need to set a timeout before checking steading states or we get bad block layouts
             setTimeout(function(){waitForSSAndSave(params, function(err, sid){
             console.warn()
             });}, 400);
             });*/
        };
        /**
         * show the state to be used as state 0
         * @param sid
         */
        $scope.showState = function (sid) {
            if (!$stateParams.sid)
                $state.transitionTo('app.genworld', { sid: sid }, { notify: false });
            $rootScope.dataloaded = false;
            $scope.enableImpSave = false;
            //we must get the state for this sid
            $scope.subscribe("genstates", function () { return [sid]; }, {
                onReady: function (sub) {
                    var myframe = GenStates.findOne({ _id: sid });
                    console.warn(sid, myframe);
                    if (!myframe)
                        return toaster.pop('warn', 'Invalid State ID');
                    //update the meta
                    $scope.curitr = myframe.block_states.length - 1;
                    $scope.curcnt = 0;
                    $scope.curState.clear();
                    $scope.curState.copy(myframe);
                    myengine.createObjects($scope.curState.block_meta.blocks);
                    myengine.updateScene(myframe.block_states[$scope.curitr]);
                    function itrScreencap(idx, list, cb) {
                        if (_.isUndefined(list[idx])) {
                            $scope.$apply(function () { $rootScope.dataloaded = true; });
                            return cb();
                        }
                        var scid = list[idx].screencapid;
                        $scope.subscribe("screencaps", function () { return [scid]; }, {
                            onReady: function (sub) {
                                var screen = ScreenCaps.findOne({ _id: scid });
                                var attachid = createButtons('stateimg', idx);
                                showImage(screen.data, 'Move #:' + idx, attachid);
                                itrScreencap(idx + 1, list, cb);
                            },
                            onStop: subErr
                        });
                    }
                    itrScreencap(0, myframe.block_states, function () {
                    });
                },
                onStop: subErr
            });
        };
        var createButtons = function (id, i) {
            var lenID = $('div').length;
            var eleDivID = 'rowdiv' + lenID; // Unique ID
            var retId = id + lenID;
            var htmlout = '';
            if (myengine.getUIVal()) {
                htmlout =
                    '<button onclick="angular.element(this).scope().getMove(' + i + ')" class="btn btn-xs btn-info"> Get JSON </button>' +
                        '<div id="' + retId + '"></div>';
            }
            else {
                htmlout =
                    '<button onclick="angular.element(this).scope().cloneMove(' + i + ')" class="btn btn-xs btn-info"> Clone Move </button>' +
                        '    ' +
                        '<button onclick="angular.element(this).scope().getMove(' + i + ')" class="btn btn-xs btn-info"> Get JSON </button>' +
                        '    ' +
                        '<button onclick="angular.element(this).scope().delMove(' + i + ')" class="btn btn-xs btn-info"> Delete Move </button>' +
                        '<div id="' + retId + '"></div>';
            }
            var attachTo = '#galleryarea';
            $('<div>').attr({
                id: eleDivID
            }).addClass('col-sm-4')
                .html(htmlout).css({ "border-bottom": '1px solid #e4eaec' }).appendTo(attachTo);
            return retId;
        };
        $scope.remState = function (sid) {
            if (sid) {
                $scope.subscribe("genstates", function () { return [sid]; }, {
                    onReady: function (sub) {
                        var myframe = GenStates.findOne({ _id: sid });
                        myframe.block_states.forEach(function (s) {
                            ScreenCaps.remove(s.screencapid);
                        });
                        GenStates.remove(sid);
                        updateTableStateParams();
                        toaster.pop('warning', 'Removed ' + sid);
                    },
                    onStop: subErr
                });
            }
        };
        var getStackCubes = function (mycube, used, cid, checkY) {
            var retStack = [];
            for (var i = 0; i < used.length; i++) {
                if (!cid || cid != used[i].prop.cid) {
                    var c = used[i];
                    if (myengine.intersectsMeshXYZ(mycube, c, checkY)) {
                        retStack.push(c);
                    }
                }
            }
            return retStack;
        };
        /*$scope.myreplay = null;
         $scope.frameid = -1;
         var showReplay = function(idx){
         var frameScene = $scope.myreplay.data.act[idx];
         frameScene.forEach(function(frame){
         var cube = cubesnamed[frame.name];
         cube.position = new BABYLON.Vector3(frame.position.x, frame.position.y, frame.position.z);
         cube.rotationQuaternion = new BABYLON.Quaternion(frame.rotation.x, frame.rotation.y, frame.rotation.z, frame.rotation.w);
         cube.isVisible = true;
         })
         };*/
        $scope.enableImpSave = false;
        $scope.cancelImport = function () {
            //must use function to apply to scope
            $scope.impFilename = null;
            $scope.enableImpSave = false;
            $scope.clearMeta();
        };
        $scope.saveImport = function (savename) {
            $rootScope.dataloaded = false;
            $scope.impFilename = null;
            $scope.enableImpSave = false;
            var cubesused = [];
            $scope.curState.block_meta.blocks.forEach(function (b) {
                cubesused.push(b.id);
            });
            cubesused = _.uniq(cubesused);
            if (!$scope.curState.block_meta.decoration) {
                //set decoration if we don't have one
                if (!$scope.opt.showImages)
                    $scope.curState.block_meta.decoration = cBlockDecor.blank;
                else {
                    if ($scope.opt.showLogos)
                        $scope.curState.block_meta.decoration = cBlockDecor.logo;
                    else
                        $scope.curState.block_meta.decoration = cBlockDecor.digit;
                }
            }
            $scope.curState.name = savename;
            if ($scope.curState.created && $scope.curState.creator && !_.isUndefined($scope.curState.public)) {
                console.warn('saveImport');
                var params = { itr: 0, startMove: null, cubesused: cubesused };
                setTimeout(function () {
                    waitForSSAndSave(params, function (err, savedsid) {
                        console.warn('saveimport wait for');
                        if (err)
                            toaster.pop('warn', err);
                        if (savedsid) {
                            $scope.curitr = $scope.curState.stateitr;
                            $scope.curcnt = 0;
                            updateTableStateParams();
                            $state.transitionTo('app.genworld', { sid: savedsid }, { notify: false });
                        }
                        $rootScope.dataloaded = true;
                    });
                }, 400);
            }
            else
                toaster.pop('error', 'Missing Creator Information');
        };
        $scope.clearMeta = function () {
            $('#galleryarea').empty();
            $scope.curState.clear();
            $scope.curcnt = -1;
            $scope.curitr = -1;
            $scope.enableUI = false;
            $scope.enableImpSave = false;
            myengine.resetWorld();
            myengine.setUI($scope.enableUI);
            $state.transitionTo('app.genworld', {}, { notify: false });
        };
        $scope.loadMeta = function () {
            if ($scope.metafilename && $scope.metafilename.length) {
                //read file
                var reader = new FileReader();
                reader.onload = function () {
                    var filedata = JSON.parse(reader.result);
                    if (filedata.blocks && filedata.blocks.length) {
                        $scope.$apply(function () {
                            $scope.curState.clear();
                            $scope.curState.block_meta = filedata;
                            myengine.createObjects($scope.curState.block_meta.blocks);
                        });
                    }
                    else
                        $scope.$apply(function () {
                            toaster.pop('warn', 'Invalid JSON META file');
                        });
                };
                reader.readAsText($scope.metafilename[0]);
            }
        };
        $scope.metaFileChanged = function (event) {
            $scope.$apply(function () {
                $scope.metafilename = event.target.files;
            });
            console.warn($scope.metafilename);
        };
        /**
         * loads a json state file with the CURRENT state iteration set to 0
         */
        $scope.loadState = function () {
            if ($scope.statefilename && $scope.statefilename.length) {
                //read file
                var reader = new FileReader();
                reader.onload = function () {
                    var filedata = JSON.parse(reader.result);
                    if (filedata.block_state && filedata.block_state.length
                        && filedata.block_meta && filedata.block_meta.blocks && filedata.block_meta.blocks.length) {
                        if (filedata.block_meta.blocks.length != filedata.block_state.length)
                            return $scope.$apply(function () {
                                toaster.pop('error', 'Block META and STATE mismatch!');
                            });
                        $scope.curState.clear();
                        $scope.curState.block_meta = filedata.block_meta;
                        $scope.curState.public = true;
                        $scope.curState.created = (new Date).getTime();
                        $scope.curState.creator = $rootScope.currentUser._id;
                        setDecorVal(filedata.block_meta.decoration);
                        console.warn($scope.curState.block_meta);
                        myengine.createObjects($scope.curState.block_meta.blocks);
                        //mung block_state
                        //filedata.block_state = mungeBlockState(filedata.block_state);
                        $scope.$apply(function () {
                            $scope.impFilename = null;
                            $scope.enableImpSave = false;
                            $scope.isgen = true;
                        });
                        var block_state = mungeBlockState(filedata.block_state);
                        myengine.updateScene({ block_state: block_state }, function () {
                            $scope.$apply(function () {
                                if (filedata.name)
                                    $scope.impFilename = filedata.name;
                                else
                                    $scope.impFilename = $scope.statefilename[0].name.toLowerCase().replace(/\.json/g, '');
                                $scope.enableImpSave = true;
                                $scope.isgen = false;
                            });
                        });
                    }
                    else
                        $scope.$apply(function () {
                            toaster.pop('warn', 'Invalid JSON STATE file');
                        });
                };
                reader.readAsText($scope.statefilename[0]);
            }
        };
        $scope.stateFileChanged = function (event) {
            $scope.$apply(function () {
                $scope.statefilename = event.target.files;
            });
            console.warn($scope.statefilename);
        };
        var setDecorVal = function (decor) {
            if (decor) {
                $scope.$apply(function () {
                    //set switches
                    switch (decor) {
                        case cBlockDecor.digit:
                            $scope.opt.showImages = true;
                            $scope.opt.showLogos = false;
                            break;
                        case cBlockDecor.logo:
                            $scope.opt.showImages = true;
                            $scope.opt.showLogos = true;
                            break;
                        case cBlockDecor.blank:
                            $scope.opt.showImages = false;
                            $scope.opt.showLogos = false;
                            break;
                    }
                });
            }
        };
        $scope.loadStates = function () {
            var self = this;
            if ($scope.statesfilename && $scope.statesfilename.length) {
                //read file
                var reader = new FileReader();
                reader.onload = function () {
                    var filedata = JSON.parse(reader.result);
                    if (filedata.block_states && filedata.block_states.length
                        && filedata.block_meta && filedata.block_meta.blocks && filedata.block_meta.blocks.length) {
                        if (filedata.block_meta.blocks.length != filedata.block_states[0].block_state.length)
                            return $scope.$apply(function () {
                                toaster.pop('error', 'Block META and STATE mismatch!');
                            });
                        $scope.curState.clear();
                        $scope.curState.block_meta = filedata.block_meta;
                        if (filedata.type)
                            $scope.curState.type = filedata.type;
                        $scope.curState.public = true;
                        $scope.curState.created = (new Date).getTime();
                        $scope.curState.creator = $rootScope.currentUser._id;
                        setDecorVal(filedata.block_meta.decoration);
                        console.warn($scope.curState.block_meta);
                        myengine.createObjects($scope.curState.block_meta.blocks);
                        //mung block_states
                        $scope.curState.block_states = mungeBlockStates(filedata.block_states);
                        $scope.$apply(function () {
                            $scope.impFilename = null;
                            $scope.enableImpSave = false;
                            $scope.isgen = true;
                        });
                        var itrFrame = function (idx, block_states, cb) {
                            if (_.isUndefined(block_states[idx])) {
                                var done = function (cb) {
                                    if (filedata.name)
                                        $scope.impFilename = filedata.name;
                                    else
                                        $scope.impFilename = $scope.statesfilename[0].name.toLowerCase().replace(/\.json/g, '');
                                    $scope.enableImpSave = true;
                                    $scope.isgen = false;
                                    cb();
                                };
                                $scope.$apply(done(cb));
                            }
                            else {
                                myengine.updateScene(block_states[idx], function () {
                                    //wait for steady state
                                    checkFnSS = setInterval(function () {
                                        if (myengine.isSteadyState) {
                                            clearInterval(checkFnSS);
                                            var sc = BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
                                                width: myengine.canvas.width, height: myengine.canvas.height
                                            }, function (b64i) {
                                                var b64img = LZString.compressToUTF16(b64i);
                                                /*console.warn('len', b64i.length, b64img.length);
                                                 console.warn('b64i', b64i);
                                                 console.warn('b64img', LZString.decompressFromUTF16(b64img));*/
                                                block_states[idx].screencap = b64img;
                                                block_states[idx].created = (new Date).getTime();
                                                var attachid = createButtons('stateimg', idx);
                                                showImage(b64img, 'Move #: ' + idx, attachid);
                                                itrFrame(idx + 1, block_states, cb);
                                            });
                                        }
                                    }, 100);
                                });
                            }
                        };
                        itrFrame(0, $scope.curState.block_states, function () { });
                    }
                    else
                        $scope.$apply(function () {
                            toaster.pop('warn', 'Invalid JSON STATE file');
                        });
                };
                reader.readAsText($scope.statesfilename[0]);
            }
        };
        $scope.statesFileChanged = function (event) {
            $scope.$apply(function () {
                $scope.statesfilename = event.target.files;
            });
            console.warn($scope.statesfilename);
        };
        var mungeBlockStates = function (bss) {
            var newbss = [];
            for (var i = 0; i < bss.length; i++) {
                var ele = { block_state: mungeBlockState(bss[i].block_state) };
                if (!_.isUndefined(bss[i].enablephysics))
                    ele['enablephysics'] = bss[i].enablephysics;
                newbss.push(ele);
            }
            return newbss;
        };
        /**
         * Transform text block state from cwic to internal block states
         * @param bs
         * @returns {Array}
         */
        var mungeBlockState = function (bs) {
            var newBS = [];
            bs.forEach(function (b) {
                var li = b.position.split(',');
                var lv = [];
                li.forEach(function (v, i) {
                    lv.push(Number(v));
                });
                if (b.rotation) {
                    var ri = b.rotation.split(',');
                    var rv = [];
                    ri.forEach(function (v, i) {
                        rv.push(Number(v));
                    });
                    newBS.push({
                        id: b.id, position: {
                            x: lv[0], y: lv[1], z: lv[2]
                        }, rotation: {
                            x: rv[0], y: rv[1], z: rv[2], w: rv[3]
                        }
                    });
                }
                else
                    newBS.push({
                        id: b.id, position: {
                            x: lv[0], y: lv[1], z: lv[2]
                        }
                    });
            });
            return newBS;
        };
        $scope.startMove = function (itr) {
            console.warn(itr);
            itr = Number(itr);
            $scope.isgen = true;
            var params = { itr: itr, startMove: $scope.startMove, cubesused: null };
            $scope.genStateN(params);
        };
        var nextItr = function (params) {
            return function (err, savedsid) {
                if (err)
                    toaster.pop('warn', err);
                if (savedsid) {
                    if (params.itr > 1) {
                        //if(params.startGen) params.startGen(params.itr - 1);
                        if (params.startMove)
                            params.startMove(params.itr - 1);
                    }
                    else {
                        $scope.curitr = 0;
                        $scope.curcnt = 0;
                        $scope.isgen = false;
                    }
                }
                else {
                    //don't iterate since we had error with previous insert
                    //which means we need to make a new init state
                    //if(params.startGen) params.startGen(params.itr);
                    if (params.startMove)
                        params.startMove(params.itr);
                }
            };
        };
        $scope.cloneMove = function (idx) {
            var prevState = _.extend({}, $scope.curState);
            $scope.curState.clear();
            $scope.curState.block_meta = prevState.block_meta;
            $scope.curState.public = true;
            $scope.curState.created = (new Date).getTime();
            $scope.curState.creator = $rootScope.currentUser._id;
            $('#galleryarea').empty();
            myengine.createObjects($scope.curState.block_meta.blocks);
            myengine.updateScene(prevState.block_states[idx], function () {
                $scope.$apply(function () {
                    if (prevState.name)
                        $scope.impFilename = prevState.name;
                    $scope.enableImpSave = true;
                });
            });
        };
        $scope.dlScene = function () {
            var tempframe = {
                _id: $scope.curState._id,
                public: $scope.curState.public, name: $scope.curState.name, created: $scope.curState.created,
                creator: $scope.curState.creator, block_meta: $scope.curState.block_meta, block_states: []
            };
            for (var idx = 0; idx < $scope.curState.block_states.length; idx++) {
                var block_state = $scope.curState.block_states[idx].block_state;
                var newblock_state = [];
                for (var i = 0; i < block_state.length; i++) {
                    var s = block_state[i];
                    var pos = '', rot = '';
                    _.each(s.position, function (v) {
                        if (pos.length)
                            pos += ',';
                        pos += fixedNumber(v);
                    });
                    _.each(s.rotation, function (v) {
                        if (rot.length)
                            rot += ',';
                        rot += fixedNumber(v);
                    });
                    newblock_state.push({ id: s.id, position: pos, rotation: rot });
                }
                var ele = { block_state: newblock_state, enablephysics: myengine.opt.hasPhysics };
                //override with blockstate physics
                if (!_.isUndefined($scope.curState.block_states[idx].enablephysics)) {
                    ele.enablephysics = $scope.curState.block_states[idx].enablephysics;
                }
                tempframe.block_states.push(ele);
            }
            var content = JSON.stringify(angular.copy(tempframe), null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_scene_' + $scope.curState._id + '.json');
        };
        $scope.getMove = function (idx) {
            var tempframe = {
                block_meta: $scope.curState.block_meta,
                block_state: []
            };
            var block_state = $scope.curState.block_states[idx].block_state;
            for (var i = 0; i < block_state.length; i++) {
                var s = block_state[i];
                var pos = '', rot = '';
                _.each(s.position, function (v) {
                    if (pos.length)
                        pos += ',';
                    pos += fixedNumber(v);
                });
                _.each(s.rotation, function (v) {
                    if (rot.length)
                        rot += ',';
                    rot += fixedNumber(v);
                });
                tempframe.block_state.push({ id: s.id, position: pos, rotation: rot });
            }
            var content = JSON.stringify(angular.copy(tempframe), null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_state_' + $scope.curState._id + '_' + idx + '.json');
        };
        $scope.delMove = function (idx) {
            console.warn('delmove');
            //var count:number = $scope.curState.block_states.length - idx; //remove to end
            var count = 1; //remove only one item
            $scope.curState.block_states.splice(idx, count);
            if ($scope.curState._id) {
                var doc = angular.copy($scope.curState);
                var id = $scope.curState._id;
                delete doc._id;
                GenStates.update({ _id: id }, { $set: doc }, function (err, num) {
                    if (err)
                        return console.warn('delmove err:', err);
                    $scope.clearMeta();
                    $scope.showState(id);
                });
            }
            else
                $scope.$apply(function () { toaster.pop('info', 'Please SAVE layout before deleting moves'); });
        };
        $scope.setCreateMode = function () {
            $scope.enableUI = true;
            myengine.setUI($scope.enableUI);
            $scope.sceneExists = false;
            $scope.createStateIdx = 0;
            $scope.curState.block_states = mungeBlockStates(defBlockState.block_states);
            $scope.curState.block_meta = angular.copy(defBlockState.block_meta);
            $scope.enabledCubes.length = 0;
            _.each($scope.curState.block_meta.blocks, function (bl) {
                $scope.enabledCubes.push({ id: bl.id, name: bl.name });
            });
            myengine.createObjects($scope.curState.block_meta.blocks);
            myengine.updateScene($scope.curState.block_states[0]);
        };
        /**save state in create mode
         * by saving position and screencap
         */
        $scope.saveState = function () {
            //wait for steady state
            checkFnSS = setInterval(function () {
                if (myengine.isSteadyState) {
                    clearInterval(checkFnSS);
                    //there should be already a view since we render one
                    var block_states = $scope.curState.block_states[$scope.createStateIdx];
                    var block_state = block_states.block_state;
                    block_state.length = 0;
                    //fill the block state for this frame with information
                    $scope.curState.block_meta.blocks.forEach(function (bl) {
                        var c = myengine.get3DCubeById(bl.id);
                        if (c.isVisible) {
                            var bs = { id: bl.id, position: { x: fixedNumber(c.position.x), y: fixedNumber(c.position.y), z: fixedNumber(c.position.z) } };
                            if (c.rotationQuaternion)
                                bs.rotation = { x: fixedNumber(c.rotationQuaternion.x), y: fixedNumber(c.rotationQuaternion.y), z: fixedNumber(c.rotationQuaternion.z), w: fixedNumber(c.rotationQuaternion.w) };
                            block_state.push(bs);
                        }
                    });
                    if (block_state.length != $scope.curState.block_meta.blocks.length)
                        return $scope.$apply(function () {
                            toaster.pop('error', 'Block META and STATE mismatch!');
                        });
                    var sc = BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
                        width: myengine.canvas.width, height: myengine.canvas.height
                    }, function (b64i) {
                        var b64img = LZString.compressToUTF16(b64i);
                        /*console.warn('len', b64i.length, b64img.length);
                         console.warn('b64i', b64i);
                         console.warn('b64img', LZString.decompressFromUTF16(b64img));*/
                        block_states.screencap = b64img;
                        block_states.created = (new Date).getTime();
                        block_states.enablephysics = $scope.opt.hasPhysics;
                        var attachid = createButtons('stateimg', $scope.createStateIdx);
                        showImage(b64img, 'Move #: ' + $scope.createStateIdx, attachid);
                    });
                    $scope.createStateIdx++;
                    //now copy the current layout to the next view incase we want to undo etc.
                    if (!$scope.curState.block_states[$scope.createStateIdx]) {
                        //empty block states place holder
                        var blockstatesph = { created: 0, screencapid: '', block_state: [] };
                        $scope.curState.block_states.push(blockstatesph);
                    }
                    var block_states = $scope.curState.block_states[$scope.createStateIdx];
                    //make a copy of the previous states into this for reset if needed
                    block_states.block_state = angular.copy(block_state);
                    $scope.$apply(function () {
                        $scope.sceneExists = true;
                    });
                }
            }, 100);
        };
        $scope.hideCube = function (cnt) {
            if (cnt) {
                var block_states = $scope.curState.block_states[$scope.createStateIdx];
                //block_states.block_state = _.filter(block_states.block_state, function(bl:iBlockState){return !(bl.id == id)});
                //$scope.enabledCubes = _.filter($scope.enabledCubes, function(bl:iBlockMetaEle){return !(bl.id == id);});
                var total = $scope.curState.block_meta.blocks.length;
                block_states.block_state.splice(total - cnt, cnt);
                $scope.curState.block_meta.blocks.splice(total - cnt, cnt);
                $scope.enabledCubes.splice(total - cnt, cnt);
                myengine.updateScene(block_states);
            }
        };
        var radian = function (degrees) {
            return degrees * Math.PI / 180;
        };
        $scope.rotCube = function (rotid, axis, deg) {
            if (rotid) {
                var rad = radian(deg);
                var c = myengine.get3DCubeById(rotid);
                if (c.isVisible)
                    c.rotate(BABYLON.Axis[axis], rad, BABYLON.Space.LOCAL);
            }
        };
        /**Reset Frame to the last saved update
         * */
        $scope.resetState = function () {
            myengine.updateScene($scope.curState.block_states[$scope.createStateIdx]);
        };
        $scope.resetCamera = function () {
            myengine.resetCamera();
        };
        /**Use existing save scene from import
         * */
        $scope.saveScene = function () {
            //set curState information since we are saving
            $scope.curState.public = true;
            $scope.curState.created = (new Date).getTime();
            $scope.curState.creator = $rootScope.currentUser._id;
            //remove the placeholder frame because its not valid
            $scope.curState.block_states.length = $scope.curState.block_states.length - 1;
            $scope.enableImpSave = true;
            $scope.sceneExists = false;
            $scope.enableUI = false;
            myengine.setUI($scope.enableUI);
        };
        /**Update Pyshics so all items will activate
         * If we turn it on then we have to save the current scene and add physics to the objects
         * */
        $scope.updatePhysics = function () {
            $scope.opt.hasPhysics = !$scope.opt.hasPhysics;
            myengine.updatePhysics();
        };
        var fixedNumber = function (x) { return Number(x.toFixed(5)); };
        var defBlockState = (new cDefBlockData()).get();
        // Start by calling the createScene function that you just finished creating
        var myengine = new mGen3DEngine.cUI3DEngine(APP_CONST.fieldsize);
        $scope.enabledCubes = [];
        $scope.opt = myengine.opt;
        $scope.opt.limStack = true; //we add a stack limit to 3d engine vars
        $scope.enableUI = false;
        myengine.createWorld();
        myengine.setUI($scope.enableUI);
        dataReady.update('world created');
    }]);
var cDefBlockData = (function () {
    function cDefBlockData() {
        this.data = {
            _id: null, 'public': true, name: 'default', created: null, creator: 'system',
            "block_states": [{ "block_state": [] }],
            "block_meta": {
                "decoration": "logo",
                "blocks": [{ "shape": null, "name": "adidas", "id": 1 }, { "shape": null, "name": "bmw", "id": 2 }, { "shape": null, "name": "burger king", "id": 3 },
                    { "shape": null, "name": "coca cola", "id": 4 }, { "shape": null, "name": "esso", "id": 5 }, { "shape": null, "name": "heineken", "id": 6 }, {
                        "shape": null, "name": "hp", "id": 7 },
                    { "shape": null, "name": "mcdonalds", "id": 8 }, { "shape": null, "name": "mercedes benz", "id": 9 }, { "shape": null, "name": "nvidia", "id": 10 },
                    { "shape": null, "name": "pepsi", "id": 11 }, { "shape": null, "name": "shell", "id": 12 }, { "shape": null, "name": "sri", "id": 13 }, { "shape": null, "name": "starbucks", "id": 14 },
                    { "shape": null, "name": "stella artois", "id": 15 }, { "shape": null, "name": "target", "id": 16 }, { "shape": null, "name": "texaco", "id": 17 },
                    { "shape": null, "name": "toyota", "id": 18 }, { "shape": null, "name": "twitter", "id": 19 }, { "shape": null, "name": "ups", "id": 20 }]
            }
        };
        var shape = { "shape_params": { "face_4": { "color": "magenta", "orientation": 1 }, "face_5": { "color": "yellow", "orientation": 1 },
                "face_6": { "color": "red", "orientation": 2 }, "face_1": { "color": "blue", "orientation": 1 },
                "face_2": { "color": "green", "orientation": 1 }, "face_3": { "color": "cyan", "orientation": 1 }, "side_length": 0.1524 },
            "type": "cube", "size": 0.5 };
        var p = -0.5;
        var i = 0;
        var z = 0;
        var zpos = [0, 0.3, 0.6, 0.9];
        for (var j = 0; j < 20; j++) {
            this.data.block_meta.blocks[j]['shape'] = angular.copy(shape);
            var pos = [(p + i * 0.3), Number(shape.shape_params.side_length), zpos[z]];
            this.data.block_states[0].block_state.push({ position: pos.join(','), id: j + 1 });
            if (i > 3) {
                i = 0;
                z++;
            }
            else
                i++;
        }
    }
    cDefBlockData.prototype.get = function () {
        return angular.copy(this.data);
    };
    return cDefBlockData;
}());
//# sourceMappingURL=gen-world-view.js.map