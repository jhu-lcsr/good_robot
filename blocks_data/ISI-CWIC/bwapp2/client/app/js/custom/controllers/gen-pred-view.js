/**========================================================
 * Module: gen-pred-view.ts
 * Created by wjwong on 9/9/15.
 =========================================================*/
/// <reference path="gen-3d-engine.ts" />
/// <reference path="../../../../../model/genstatesdb.ts" />
/// <reference path="../../../../../model/screencapdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../public/vendor/babylonjs/babylon.2.2.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />
angular.module('app.generate').controller('genPredCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', 'toaster', 'APP_CONST', 'AppUtils', function ($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, toaster, APP_CONST, apputils) {
        "use strict";
        var mult = 100; //position multiplier for int random
        var checkFnSS; //store steady state check
        $scope.curState = new apputils.cCurrentState();
        var dataReady = new apputils.cDataReady(0, function () {
            $rootScope.dataloaded = true;
        });
        $scope.resetWorld = function () {
            //resetworld 
            myengine.resetWorld();
        };
        /*var showFrame = function(state:iBlockStates, cb?: ()=>void){
          $scope.resetWorld();
          setTimeout(function(){
            if(state.block_state){
              state.block_state.forEach(function(frame){
                var c = myengine.get3DCubeById(frame.id);
                c.position = new BABYLON.Vector3(frame.position['x'], frame.position['y'], frame.position['z']);
                if(frame.rotation)
                  c.rotationQuaternion = new BABYLON.Quaternion(frame.rotation['x'], frame.rotation['y'], frame.rotation['z'], frame.rotation['w']);
                c.isVisible = true;
                if(myengine.hasPhysics) c.setPhysicsState({
                  impostor: BABYLON.PhysicsEngine.BoxImpostor,
                  move: true,
                  mass: 5, //c.boxsize,
                  friction: myengine.fric,
                  restitution: myengine.rest
                });
              });
            }
            else $scope.$apply(function(){toaster.pop('error', 'Missing BLOCK_STATE')});
            if(cb) cb();
          }, 100);
        };*/
        var createLayout = function (id, i, attachID) {
            var lenID = $('div').length;
            var eleDivID = 'rowdiv' + lenID; // Unique ID
            var retId = id + lenID;
            var htmlout = 
            //'<button onclick="angular.element(this).scope().showPrediction('+i+')" class="btn btn-xs btn-info"> View </button>'+
            '<div id="' + retId + '" onclick="angular.element(this).scope().showPrediction(' + i + ')"></div>';
            var attachTo = '#galleryarea';
            if (attachID)
                attachTo = '#' + attachID;
            $('<div>').attr({
                id: eleDivID
            }).addClass('col-sm-4')
                .html(htmlout).css({ "border-bottom": '1px solid #e4eaec' }).appendTo(attachTo);
            return retId;
        };
        var showImage = function (b64i, text, attachID, scale) {
            var b64img = LZString.decompressFromUTF16(b64i);
            var eleDivID = 'div' + $('div').length; // Unique ID
            var eleImgID = 'img' + $('img').length; // Unique ID
            //var eleLabelID:string = 'h4' + $('h4').length; // Unique ID
            var htmlout = '';
            if (text)
                htmlout += '<b>' + text + '</b><br>';
            var width = myengine.canvas.width * ((scale) ? scale : 1);
            var height = myengine.canvas.height * ((scale) ? scale : 1);
            htmlout += '<img id="' + eleImgID + '" style="width:' + width + 'px;height:' + height + 'px"></img>';
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
        $scope.clearMeta = function () {
            $('#galleryarea').empty();
            $scope.curState.clear();
            $state.transitionTo('app.genpred', {}, { reload: true, notify: true });
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
        $scope.statesFileChanged = function (event) {
            $scope.$apply(function () { $scope.statesfilename = event.target.files; });
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
                li.forEach(function (v, i) { lv.push(Number(v)); });
                if (b.rotation) {
                    var ri = b.rotation.split(',');
                    var rv = [];
                    ri.forEach(function (v, i) {
                        rv.push(Number(v));
                    });
                    newBS.push({ id: b.id, position: {
                            x: lv[0], y: lv[1], z: lv[2]
                        }, rotation: {
                            x: rv[0], y: rv[1], z: rv[2], w: rv[3]
                        } });
                }
                else
                    newBS.push({ id: b.id, position: {
                            x: lv[0], y: lv[1], z: lv[2]
                        } });
            });
            return newBS;
        };
        $scope.loadStates = function () {
            if ($scope.statesfilename && $scope.statesfilename.length) {
                //read file
                var reader = new FileReader();
                reader.onload = function () {
                    var filedata = JSON.parse(reader.result);
                    var diffbm = JSON.parse(reader.result).block_meta; //store a copy of the blockmeta for use in diff view
                    if (filedata.block_meta && filedata.block_meta.blocks && filedata.block_meta.blocks.length
                        && filedata.predictions && filedata.predictions.length) {
                        $rootScope.dataloaded = false;
                        $scope.curState.clear();
                        $scope.curState.block_meta = _.extend({}, filedata.block_meta);
                        //create a copy of cubes it for gold or predicted view
                        _.each(diffbm.blocks, function (b) {
                            var bl = _.extend({}, b);
                            bl.id = Number(bl.id) + 100; //stagger by 100 in the id
                            _.each(bl.shape.shape_params, function (v, k) {
                                if (v.color)
                                    bl.shape.shape_params[k].color = 'cyan';
                            });
                            $scope.curState.block_meta.blocks.push(bl); //save this copy
                        });
                        $scope.curState.public = true;
                        $scope.curState.created = (new Date).getTime();
                        $scope.curState.creator = $rootScope.currentUser._id;
                        $scope.curState.name = $scope.statesfilename[0].name;
                        $scope.predictions = filedata.predictions;
                        setDecorVal(filedata.block_meta.decoration);
                        myengine.createObjects($scope.curState.block_meta.blocks);
                        //$scope.showPrediction(0);
                        $scope.diffPredictions = [];
                        procDiff(0, $scope.diffPredictions, function () {
                            if ($scope.diffPredictions.length)
                                renderGallery(0, function () {
                                    $scope.$apply(function () {
                                        $rootScope.dataloaded = true;
                                    });
                                });
                        });
                    }
                    else
                        $scope.$apply(function () { toaster.pop('warn', 'Invalid JSON STATE file'); });
                };
                reader.readAsText($scope.statesfilename[0]);
            }
        };
        var diffPrediction = function (idx) {
            var p = $scope.predictions[idx];
            var goldbs = mungeBlockState(p['gold_state'].block_state);
            var predbs = mungeBlockState(p['predicted_state'].block_state);
            ;
            //create an associative array of position and id - then we will remove each predicted cube that ovdr lap with gold
            var predlist = {};
            _.each(predbs, function (p) {
                predlist[p.id] = p.position;
            });
            //now iterate gold blocks list and start removing prediction blocks when they cover each other
            //whats left in predlist and not found from gold is the non overlap 
            function remOverlap(idx, gbs, predl, uniqs, cb) {
                if (_.isUndefined(gbs[idx]))
                    return cb();
                var bFound = false;
                _.each(predl, function (p, k) {
                    //only match one block to one block not one gbs block to more than one predicted block
                    if (!bFound) {
                        var subp = subtractPos(gbs[idx].position, p);
                        if (isZeroPos(subp)) {
                            delete predl[k];
                            bFound = true;
                        }
                    }
                });
                if (!bFound) {
                    uniqs.push(gbs[idx]);
                }
                remOverlap(idx + 1, gbs, predl, uniqs, cb);
            }
            var uniqlist = [];
            remOverlap(0, goldbs, predlist, uniqlist, function () {
                //save whats left of blocks in predicted view
                _.each(predlist, function (p, k) {
                    var val = { id: Number(k) + 100, position: p };
                    uniqlist.push(val);
                });
            });
            return uniqlist;
        };
        var subtractPos = function (a, b) {
            var retv = _.extend({}, a);
            _.each(retv, function (v, k) {
                retv[k] = retv[k] - b[k];
            });
            return retv;
        };
        var isZeroPos = function (p) {
            var isz = true;
            _.each(p, function (v) {
                if (v < -0.001 || v > 0.001)
                    isz = false;
            });
            return isz;
        };
        var pidx = ['start_state', 'gold_state', 'predicted_state', 'diff_state'];
        var procDiff = function (idx, retDiff, cb) {
            if (_.isUndefined($scope.predictions[idx]))
                return cb();
            var rawP = $scope.predictions[idx];
            var pred = {
                predicted_state: null,
                utterance: (rawP.utterance) ? rawP.utterance : null,
                gold_state: null,
                start_state: null,
                diff_state: null
            };
            _.each(pidx, function (aid) {
                if (aid !== 'diff_state')
                    pred[aid] = { block_state: mungeBlockState(rawP[aid].block_state) };
            });
            pred.diff_state = { block_state: diffPrediction(idx) };
            retDiff.push(pred);
            procDiff(idx + 1, retDiff, cb);
        };
        var renderGallery = function (idx, cb) {
            if (_.isUndefined($scope.diffPredictions[idx])) {
                $scope.$apply(function () { $scope.isgen = false; });
                return cb();
            }
            $scope.isgen = true;
            var k = 'diff_state';
            var dp = $scope.diffPredictions[idx];
            var block_state = dp[k];
            myengine.updateScene(block_state, function () {
                //wait for steady state
                checkFnSS = setInterval(function () {
                    if (myengine.isSteadyState) {
                        clearInterval(checkFnSS);
                        var sc = BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
                            width: myengine.canvas.width, height: myengine.canvas.height
                        }, function (b64i) {
                            var b64img = LZString.compressToUTF16(b64i);
                            var attid = createLayout('diff', idx);
                            var utt = idx + ' ';
                            if (dp.utterance.length && _.isArray(dp.utterance[0]))
                                utt += dp.utterance[0].join(' ').toUpperCase();
                            showImage(b64img, utt['trunc'](38, true), attid, 0.7);
                            renderGallery(idx + 1, cb);
                        });
                    }
                }, 100);
            });
        };
        $scope.showPrediction = function (i) {
            var idx = Number(i); //ensure text string turns into number
            $scope.clearPrediction();
            $scope.isgen = true;
            $scope.curitr = idx;
            $scope.textitr = idx;
            var pred = $scope.diffPredictions[i];
            $scope.utterance = '';
            if (_.isArray(pred.utterance) && _.isArray(pred.utterance[0])) {
                _.each(pred.utterance, function (s) {
                    $scope.utterance += s.join(' ');
                });
                $scope.utterance = $scope.utterance.toUpperCase();
            }
            else
                toaster.pop('error', 'Missing Utterance string[][]');
            renderPrediction(pred);
        };
        $scope.clearPrediction = function () {
            $scope.curitr = undefined;
            $scope.textitr = undefined;
            $scope.utterance = undefined;
            _.each(pidx, function (aid) {
                $('#' + aid).empty();
            });
        };
        var renderPrediction = function (pred) {
            if (_.isUndefined(pred))
                return;
            function itrFrame(idx, idxlist, pred, cb) {
                if (_.isUndefined(idxlist[idx]))
                    return cb();
                var k = idxlist[idx];
                var block_state = pred[k];
                myengine.updateScene(block_state, function () {
                    //wait for steady state
                    checkFnSS = setInterval(function () {
                        if (myengine.isSteadyState) {
                            clearInterval(checkFnSS);
                            var sc = BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
                                width: myengine.canvas.width, height: myengine.canvas.height
                            }, function (b64i) {
                                var b64img = LZString.compressToUTF16(b64i);
                                //block_state.screencap = b64img;
                                //block_state.created = (new Date).getTime();
                                //var attachid:string = createButtons('stateimg', idx);
                                showImage(b64img, k.toUpperCase().replace(/_/g, ' '), k);
                                itrFrame(idx + 1, idxlist, pred, cb);
                            });
                        }
                    }, 100);
                });
            }
            itrFrame(0, pidx, pred, function () {
                $scope.$apply(function () {
                    $scope.isgen = false;
                });
            });
        };
        // Start by calling the createScene function that you just finished creating
        var myengine = new mGen3DEngine.c3DEngine(APP_CONST.fieldsize);
        $scope.opt = myengine.opt;
        $scope.opt.limStack = true; //we add a stack limit to 3d engine vars
        $scope.isgen = false;
        myengine.createWorld();
        dataReady.update('world created');
    }]);
//# sourceMappingURL=gen-pred-view.js.map