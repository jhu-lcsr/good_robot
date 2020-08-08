/**
 * Created by wjwong on 12/16/15.
 */
/// <reference path="gen-3d-engine.ts" />
/// <reference path="../../../../../model/genexpsdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />
angular.module('app.generate').controller('genSimpExpCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', 'toaster', 'APP_CONST', 'DTOptionsBuilder', 'AppUtils', '$reactive', function ($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, toaster, APP_CONST, DTOptionsBuilder, apputils, $reactive) {
        "use strict";
        $reactive(this).attach($scope);
        //if user is not logged and during this view and we log in then fire a reload
        Accounts.onLogin(function (user) { $state.reload(); });
        $scope.isGuest = $rootScope.isRole(Meteor.user(), 'guest');
        var mult = 100; //position multiplier for int random
        //subscription error for onStop;
        var subErr = function (err) { if (err)
            console.warn("err:", arguments, err); return; };
        $scope.dtOptionsBootstrap = DTOptionsBuilder.newOptions()
            .withBootstrap()
            .withBootstrapOptions({
            pagination: {
                classes: {
                    ul: 'pagination pagination-sm'
                }
            }
        });
        $scope.dtOptionsAvail = _.extend({}, $scope.dtOptionsBootstrap, {
            "lengthMenu": [[5], [5]],
            "language": { "paginate": { "next": '▶', "previous": '◀' } },
            "dom": '<"pull-left"f><"pull-right"i>rt<"pull-left"p>'
        });
        if ($scope.isGuest)
            $scope.dtOptionsAvail.order = [[2, "desc"]];
        else
            $scope.dtOptionsAvail.order = [[3, "desc"]];
        $scope.curState = new apputils.cCurrentState();
        setTimeout(function () {
            $scope.subscribe("genexps", function () { }, {
                onReady: function (sid) { dataReady.update('genexps'); },
                onStop: subErr
            });
        }, 10);
        var dataReady = new apputils.cDataReady(1, function () {
            updateAvailExp();
            if ($stateParams.sid) {
                $scope.showState($stateParams.sid);
            }
            else
                $rootScope.dataloaded = true;
        });
        var updateAvailExp = function () {
            $scope.availExp = GenExps.find({}, { sort: { "_id": 1 } }).fetch();
        };
        $scope.resetWorld = function () {
            //resetworld 
            myengine.resetWorld();
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
         * show the state to be used as state 0
         * @param sid
         */
        $scope.showState = function (sid) {
            if (!$stateParams.sid)
                $state.transitionTo('app.gensimpexp', { sid: sid }, { notify: false });
            $rootScope.dataloaded = false;
            $scope.enableImpSave = false;
            $scope.isExp = true;
            //we must get the state for this sid
            $scope.subscribe("genexps", function () { return [sid]; }, {
                onReady: function (sub) {
                    var myframe = GenExps.findOne({ _id: sid });
                    if (!myframe) {
                        $rootScope.dataloaded = true;
                        toaster.pop('warn', 'Invalid State ID');
                        $state.transitionTo('app.gensimpexp', {}, { notify: false });
                        return;
                    }
                    //update the meta
                    $scope.curState.clear();
                    $scope.curState.copy(myframe);
                    $scope.utterance = $scope.curState.utterance.join(' ').toUpperCase();
                    setDecorVal($scope.curState.block_meta.decoration);
                    myengine.createObjects($scope.curState.block_meta.blocks);
                    myengine.updateScene({ block_state: myframe.block_state });
                    $scope.$apply(function () { $rootScope.dataloaded = true; });
                },
                onStop: subErr
            });
        };
        $scope.reset = function () {
            myengine.createObjects($scope.curState.block_meta.blocks);
            myengine.updateScene({ block_state: $scope.curState.block_state });
        };
        $scope.remState = function (sid) {
            if (sid) {
                GenExps.remove(sid);
                updateAvailExp();
                toaster.pop('warning', 'Removed ' + sid);
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
        $scope.enableImpSave = false;
        $scope.cancelImport = function () {
            //must use function to apply to scope
            $scope.impFilename = null;
            $scope.enableImpSave = false;
            $scope.curState.clear();
            $scope.resetWorld();
        };
        $scope.saveImport = function (savename, isMulti, cb) {
            $rootScope.dataloaded = false;
            $scope.impFilename = null;
            $scope.enableImpSave = false;
            $scope.curState.name = savename;
            setTimeout(function () {
                var doc = angular.copy($scope.curState);
                GenExps.insert(doc, function (err, id) {
                    if (err)
                        toaster.pop('error', 'Save Import error: ', err.message);
                    if (!isMulti) {
                        $scope.curState._id = id;
                        $rootScope.dataloaded = true;
                        updateAvailExp();
                        $state.go('app.gensimpexp', { sid: id }, { reload: true, notify: true });
                    }
                    if (cb)
                        cb();
                    //$state.transitionTo('app.gensimpexp', {sid: val[0]._id}, {notify: false});
                });
            }, 400);
        };
        $scope.clearMeta = function () {
            $('#galleryarea').empty();
            $scope.curState.clear();
            $state.transitionTo('app.gensimpexp', {}, { notify: false });
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
            if ($scope.statesfilename && $scope.statesfilename.length) {
                if ($scope.statesfilename.length > 1) {
                    var reader = new FileReader();
                    var procFiles = function (idx, files, cb) {
                        if (_.isUndefined(files[idx]))
                            return cb();
                        reader.onload = function () {
                            var filedata = JSON.parse(reader.result);
                            var validKeys = ['block_meta', 'block_state', 'name', 'utterance'];
                            var resValidKeys = apputils.isValidKeys(filedata, validKeys);
                            if (resValidKeys.ret && filedata.block_state.length
                                && filedata.block_meta.blocks && filedata.block_meta.blocks.length) {
                                if (filedata.block_meta.blocks.length != filedata.block_state.length)
                                    return $scope.$apply(function () {
                                        toaster.pop('error', 'Block META and STATE mismatch! ' + files[idx]);
                                        procFiles(idx + 1, files, cb);
                                    });
                                $scope.curState.clear();
                                $scope.curState.block_meta = filedata.block_meta;
                                $scope.curState.public = true;
                                $scope.curState.created = (new Date).getTime();
                                $scope.curState.creator = $rootScope.currentUser._id;
                                $scope.curState.utterance = filedata.utterance;
                                $scope.utterance = filedata.utterance.join(' ').toUpperCase()['trunc'](48, true);
                                setDecorVal(filedata.block_meta.decoration);
                                $scope.curState.block_state = mungeBlockState(filedata.block_state);
                                var savename = files[idx]['name'].toLowerCase().replace(/\.json/g, '');
                                $scope.saveImport(savename, true, function () {
                                    toaster.pop('info', 'Saved ' + savename);
                                    procFiles(idx + 1, files, cb);
                                });
                            }
                            else
                                $scope.$apply(function () {
                                    toaster.pop('warn', 'Invalid JSON ' + files[idx], JSON.stringify(resValidKeys.err));
                                    procFiles(idx + 1, files, cb);
                                });
                        };
                        reader.readAsText(files[idx]);
                    };
                    procFiles(0, $scope.statesfilename, function () {
                        $rootScope.dataloaded = true;
                        $scope.curState.clear();
                        updateAvailExp();
                        $state.go('app.gensimpexp', {}, { reload: true, notify: true });
                    });
                }
                else {
                    $scope.isExp = false;
                    //read file
                    var reader = new FileReader();
                    reader.onload = function () {
                        var filedata = JSON.parse(reader.result);
                        var validKeys = ['block_meta', 'block_state', 'name', 'utterance'];
                        var resValidKeys = apputils.isValidKeys(filedata, validKeys);
                        if (resValidKeys.ret && filedata.block_state.length
                            && filedata.block_meta.blocks && filedata.block_meta.blocks.length) {
                            if (filedata.block_meta.blocks.length != filedata.block_state.length)
                                return $scope.$apply(function () {
                                    toaster.pop('error', 'Block META and STATE mismatch!');
                                });
                            $scope.curState.clear();
                            $scope.curState.block_meta = filedata.block_meta;
                            $scope.curState.public = true;
                            $scope.curState.created = (new Date).getTime();
                            $scope.curState.creator = $rootScope.currentUser._id;
                            $scope.curState.utterance = filedata.utterance;
                            $scope.utterance = filedata.utterance.join(' ').toUpperCase()['trunc'](48, true);
                            setDecorVal(filedata.block_meta.decoration);
                            myengine.createObjects($scope.curState.block_meta.blocks);
                            //mung block_states
                            $scope.curState.block_state = mungeBlockState(filedata.block_state);
                            $scope.$apply(function () {
                                $scope.impFilename = null;
                                $scope.enableImpSave = false;
                                $scope.isgen = true;
                            });
                            myengine.updateScene({ block_state: $scope.curState.block_state }, function () {
                                //wait for steady state
                                checkFnSS = setInterval(function () {
                                    if (myengine.isSteadyState) {
                                        clearInterval(checkFnSS);
                                        $scope.$apply(function () {
                                            if (filedata.name)
                                                $scope.impFilename = filedata.name;
                                            else
                                                $scope.impFilename = $scope.statesfilename[0].name.toLowerCase().replace(/\.json/g, '');
                                            $scope.enableImpSave = true;
                                            $scope.isgen = false;
                                        });
                                    }
                                }, 100);
                            });
                        }
                        else
                            $scope.$apply(function () {
                                toaster.pop('warn', 'Invalid JSON STATE file', JSON.stringify(resValidKeys.err));
                            });
                    };
                    reader.readAsText($scope.statesfilename[0]);
                }
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
                newbss.push({ block_state: mungeBlockState(bss[i].block_state) });
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
                if (typeof b.position != "string")
                    $scope.$apply(function () { toaster.pop('error', "Invalid: position should be a string list of coordinates"); });
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
        $scope.dlScene = function (notes) {
            var tempframe = {
                /*_id: $scope.curState._id,
                public: $scope.curState.public,
                created: $scope.curState.created,
                creator: $scope.curState.creator,*/
                start_id: $scope.curState._id,
                name: $scope.curState.name,
                block_meta: null,
                block_state: null,
                utterance: $scope.curState.utterance,
                notes: notes
            };
            var block_state = $scope.curState.block_state;
            var newblock_state = [];
            var cubesused = [];
            $scope.curState.block_meta.blocks.forEach(function (b) {
                cubesused.push(b.id);
            });
            cubesused = _.uniq(cubesused);
            var isValid = true;
            var max = APP_CONST.fieldsize / 2 + 0.1; //give it a little wiggle room
            var min = -max;
            var frame = [];
            var meta = { blocks: [] };
            cubesused.forEach(function (cid) {
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
                        console.warn('Out', c.position.x - c.boxsize / 2, c.position.x + c.boxsize / 2, c.position.z - c.boxsize / 2, c.position.z + c.boxsize / 2, cid, c);
                    }
                }
            });
            if (!isValid) {
                toaster.pop('error', 'Cube(s) Out of Bounds!');
                return false;
            }
            for (var i = 0; i < frame.length; i++) {
                var s = frame[i];
                var pos = '', rot = '';
                _.each(s.position, function (v) {
                    if (pos.length)
                        pos += ',';
                    pos += v;
                });
                _.each(s.rotation, function (v) {
                    if (rot.length)
                        rot += ',';
                    rot += v;
                });
                if (rot.length)
                    newblock_state.push({ id: s.id, position: pos, rotation: rot });
                else
                    newblock_state.push({ id: s.id, position: pos });
            }
            tempframe.block_state = newblock_state;
            tempframe.block_meta = meta;
            var content = JSON.stringify(tempframe, null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_scene_' + $scope.curState._id + '.json');
        };
        // Start by calling the createScene function that you just finished creating
        var myengine = new mGen3DEngine.cUI3DEngine(APP_CONST.fieldsize);
        myengine.opt.enableUI = true;
        $scope.opt = myengine.opt;
        $scope.opt.limStack = true; //we add a stack limit to 3d engine vars
        $scope.isExp = true; //all work is consider experiment view unless we import a state
        myengine.createWorld();
        dataReady.update('world created');
    }]);
//# sourceMappingURL=gen-simpexp-view.js.map