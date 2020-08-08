/**========================================================
 * Module: gen-cmdtask-view
 * Created by wjwong on 2/10/16.
 =========================================================*/
/// <reference path="gen-3d-engine.ts" />
/// <reference path="../../../../../model/gencmdjobsdb.ts" />
/// <reference path="../../../../../model/gencmdsdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/jquery/jquery.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />
var eCmdPhase;
(function (eCmdPhase) {
    eCmdPhase[eCmdPhase["VIEW"] = 0] = "VIEW";
    eCmdPhase[eCmdPhase["CMD"] = 1] = "CMD";
    eCmdPhase[eCmdPhase["RATE"] = 2] = "RATE";
    eCmdPhase[eCmdPhase["FIX"] = 3] = "FIX";
})(eCmdPhase || (eCmdPhase = {}));
;
angular.module('app.generate').controller('genCmdTaskCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', '$reactive', 'ngDialog', 'toaster', 'APP_CONST', 'AppUtils', '$filter', 'deviceDetector', function ($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, $reactive, ngDialog, toaster, APP_CONST, apputils, $filter, devDetect) {
        "use strict";
        $reactive(this).attach($scope);
        //subscription error for onStop;
        var subErr = function (err) { if (err)
            console.warn("err:", arguments, err); return; };
        $scope.date = (new Date()).getTime();
        $scope.opt = { bAgreed: true, repvalidlist: [mGenCmdJobs.eRepValid[0], mGenCmdJobs.eRepValid[1], mGenCmdJobs.eRepValid[2]], repvalid: '', isValidBrowser: (devDetect.browser.toLowerCase() === 'chrome'), command: '', viewModeCmd: '', viewIdx: -1, isBaseView: false };
        $scope.subscribe("gencmds", function () { }, {
            onReady: function (sub) {
                dataReady.update('gencmds');
            },
            onStop: subErr
        });
        $scope.subscribe("gencmdjobs", function () { }, {
            onReady: function (sub) {
                dataReady.update('gencmdjobs');
            },
            onStop: subErr
        });
        $scope.isOpenDir = true;
        $scope.taskdata;
        $scope.taskidx = -1;
        $scope.maxtask = -1;
        $scope.cmdlist = null;
        $scope.cmdele = null;
        $scope.cmdphase = eCmdPhase.CMD;
        var dataReady = new apputils.cDataReady(2, function () {
            var isAdminUser = ($rootScope.currentUser) ? $rootScope.isRole($rootScope.currentUser, 'admin') : false;
            if ($stateParams.report && !isAdminUser) {
                $rootScope.dataloaded = true;
                return;
            }
            if ($stateParams.taskId) {
                $scope.taskdata = GenCmdJobs.findOne($stateParams.taskId);
                if (!$scope.taskdata) {
                    $rootScope.dataloaded = true;
                    $scope.assignmentId = null;
                    return;
                }
                $scope = _.extend($scope, $stateParams);
                if ($scope.turkSubmitTo)
                    $scope.submitTo = $scope.turkSubmitTo + '/mturk/externalSubmit';
                if ($scope.workerId === 'EXAMPLE')
                    $scope.submitter = true;
                if (!$scope.assignmentId && !$stateParams.report && !$stateParams.json) {
                    $rootScope.dataloaded = true;
                    return;
                }
                if ($scope.hitId) {
                    //load hit
                    $scope.hitdata = GenCmdJobs.findOne('H_' + $scope.hitId);
                    if (!$scope.hitdata) {
                        $rootScope.dataloaded = true;
                        $scope.assignmentId = null;
                        return;
                    }
                    if ($scope.hitdata && $scope.hitdata.submitted && $scope.workerId && $scope.workerId !== 'EXAMPLE') {
                        var subfound = _.findWhere($scope.hitdata.submitted, { name: $scope.workerId });
                        if (!_.isUndefined(subfound)) {
                            //worker already submitted
                            $scope.submitter = subfound;
                        }
                    }
                }
                var cmdid = $scope.taskdata.cmdid;
                if (cmdid) {
                    $scope.subscribe("gencmds", function () { return [cmdid]; }, {
                        onReady: function (sub) {
                            $scope.curState = GenCmds.findOne(cmdid);
                            //console.warn('curState',$scope.curState);
                            if ($stateParams.report) {
                                $scope.report = $stateParams.report;
                                if ($scope.submitter.valid)
                                    $scope.opt.repvalid = $scope.submitter.valid;
                                else
                                    $scope.opt.repvalid = 'tbd';
                            }
                            $scope.maxtask = $scope.taskdata.antcnt;
                            $scope.taskidx = 0;
                            if ($scope.hitdata && $scope.hitdata.cmdlist && $scope.hitdata.cmdlist[$scope.workerId]) {
                                //we have hit data lets fast forward to the correct item to work on
                                //assume that we fill notes from pass 1 then pass 2 etc. there are no holes in the list
                                var mynotes = $scope.hitdata.cmdlist[$scope.workerId];
                                _.each(mynotes, function (i) {
                                    if (i && i.send && i.send.input.length)
                                        $scope.taskidx++;
                                });
                            }
                            if ($scope.taskidx || $scope.submitter)
                                $scope.opt.bAgreed = true;
                            renderTask($scope.taskidx);
                            if ($scope.submitter)
                                $scope.viewCmd($scope.taskidx - 1);
                            $scope.logolist = [];
                            _.each($scope.curState.block_meta.blocks, function (b) {
                                $scope.logolist.push({ name: b.name, imgref: "img/textures/logos/" + b.name.replace(/ /g, '') + '.png' });
                            });
                            /*Meteor.call('mturkReviewableHITs', {hid: $scope.hitId},  function(err, resp){
                             console.warn(err,resp);
                             })*/
                        },
                        onStop: subErr
                    });
                }
                else
                    toaster.pop('error', 'Missing Command ID');
            }
        });
        /*var renderReport = function(idx:number){
          if(_.isUndefined($scope.hitdata.cmdlist[$scope.workerId][idx])){ //stop at where the worker notes stop
            $rootScope.dataloaded = true;
            return;
          }
          if($scope.taskdata.tasktype == 'action'){
            var aidx:number = $scope.taskdata.idxlist[idx][0];
            var bidx:number = $scope.taskdata.idxlist[idx][1];
            $('#statea'+idx).empty();
            $('#stateb'+idx).empty();
            var scids = [$scope.curState.block_states[aidx].screencapid, $scope.curState.block_states[bidx].screencapid];
            $scope.subscribe('screencaps', ()=>{return [scids]}, {
              onReady: function(sub){
                var screena:iScreenCaps = ScreenCaps.findOne(scids[0]);
                var screenb:iScreenCaps = ScreenCaps.findOne(scids[1]);
                showImage(screena.data, 'Before', null, 'statea'+idx);
                showImage(screenb.data, 'After', null, 'stateb'+idx);
                renderReport(idx+1);
              },
              onStop: subErr
            });
          }
        };*/
        var renderTask = function (tidx) {
            if ($scope.taskidx != 0)
                $scope.isOpenDir = false;
            else
                $scope.isOpenDir = true;
            //create the annotations
            if ($scope.hitdata) {
                if (!$scope.hitdata.cmdlist)
                    $scope.hitdata.cmdlist = {};
                if (!$scope.hitdata.cmdlist[$scope.workerId])
                    $scope.hitdata.cmdlist[$scope.workerId] = {};
                if (!$scope.hitdata.cmdlist[$scope.workerId]) {
                    $scope.hitdata.cmdlist[$scope.workerId] = [];
                    for (var i = 0; i < $scope.taskdata.antcnt; i++)
                        $scope.hitdata.cmdlist[$scope.workerId].push(null);
                }
                $scope.cmdlist = $scope.hitdata.cmdlist[$scope.workerId];
            }
            else {
                $scope.cmdlist = null;
            }
            myengine.createObjects($scope.curState.block_meta.blocks);
            var states = null;
            var cidx = tidx - 1; //content idx is 1 less than tidx - so to see [0] in cmdlist you need to provide 1
            if (tidx && $scope.cmdlist[cidx] && $scope.cmdlist[cidx].send
                && $scope.cmdlist[cidx].send.world.length) {
                if ($scope.cmdlist[cidx].fix && $scope.cmdlist[cidx].fix.world.length) {
                    //show the previous command state
                    states = convCmdToState($scope.cmdlist[cidx].send, $scope.cmdlist[cidx].fix);
                }
                else if ($scope.cmdlist[cidx].recv && $scope.cmdlist[cidx].recv.world && $scope.cmdlist[cidx].recv.world.length) {
                    //show the previous command state
                    states = convCmdToState($scope.cmdlist[cidx].send, $scope.cmdlist[cidx].recv);
                }
            }
            if (states == null)
                states = { block_state: $scope.curState.block_state };
            //no move command from this user so far
            myengine.updateScene(states, function () {
                $scope.$apply(function () { $rootScope.dataloaded = true; });
            });
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
                if(frame['showMoved']) c['showMoved'] = true;
                else c['showMoved'] = false;
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
        $scope.resetWorld = function () {
            //resetworld 
            myengine.resetWorld();
        };
        var convCmdToState = function (basecmd, deltacmd) {
            var states = { block_state: [] };
            var bs = {};
            //fill all cubes
            _.each(basecmd.world, function (l) {
                var pos = { x: l.loc[0], y: l.loc[1], z: l.loc[2] };
                bs[l.id] = { id: l.id, position: pos };
                bs[l.id]['showMoved'] = false;
            });
            if (deltacmd && !$scope.opt.isBaseView) {
                //now fill delta
                _.each(deltacmd.world, function (l) {
                    var pos = { x: l.loc[0], y: l.loc[1], z: l.loc[2] };
                    bs[l.id] = { id: l.id, position: pos };
                    bs[l.id]['showMoved'] = true;
                });
            }
            _.each(bs, function (s) {
                states.block_state.push(s);
            });
            return states;
        };
        /*var showImage = function(b64i:string, title:string, caption:string, attachID:string){
          if(!attachID) return console.warn('Missing dom attach id');
          var canvas = {width: 480, height: 360};
          var b64img:string = LZString.decompressFromUTF16(b64i);
      
          var eleDivID:string = 'div' + $('div').length; // Unique ID
          var eleImgID:string = 'img' + $('img').length; // Unique ID
          var eleLabelID:string = 'label' + $('label').length; // Unique ID
          var htmlout = '<img id="'+eleImgID+'" style="width:'+canvas.width+'px;height:'+canvas.height+'px"></img>';
          if(title) htmlout = '<h3>'+title+'</h3>' + htmlout;
          if(caption) htmlout += '<label id="'+eleLabelID+'" class="mb">'+caption+'</label>';
          $('<div>').attr({
            id: eleDivID
          }).addClass('col-sm-12')
            .html(htmlout).css({}).appendTo('#'+attachID);
      
          var img:HTMLImageElement = <HTMLImageElement>document.getElementById(eleImgID); // Use the created element
          img.src = b64img;
        };
        
        var capScreen = function(){
          BABYLON.Tools.CreateScreenshot(myengine.engine, myengine.camera, {
            width: myengine.canvas.width,
            height: myengine.canvas.height
          }, function (b64i:string) {
            var b64img:string = LZString.compressToUTF16(b64i);
            showImage(b64img, 'Before', null, 'beforeimg');
          });
        };*/
        $scope.remCmd = function (idx) {
            $scope.taskidx = idx;
            delete $scope.cmdlist[idx]; //remove item from list
            delete $scope.hitdata.timed[$scope.workerId][idx]; //erase previous time
            var setdata = {};
            setdata['cmdlist.' + $scope.workerId] = $scope.hitdata.cmdlist[$scope.workerId];
            setdata['timed.' + $scope.workerId] = $scope.hitdata.timed[$scope.workerId];
            //console.warn('setdata', setdata);
            GenCmdJobs.update({ _id: $scope.hitdata._id }, {
                $set: setdata
            }, function (err, ret) {
                if (err)
                    return toaster.pop('error', err.reason);
                renderTask($scope.taskidx);
            });
        };
        $scope.viewCmd = function (idx) {
            if (idx > -1) {
                $scope.cmdphase = eCmdPhase.VIEW;
                $scope.opt.viewModeCmd = $scope.cmdlist[idx].send.input;
                $scope.opt.viewIdx = idx;
                renderTask(idx + 1);
            }
            else {
                $scope.cmdphase = eCmdPhase.CMD;
                myengine.opt.enableUI = false; //prevent UI in case its a fix transition
                $scope.cmdele = null;
                $scope.opt.viewModeCmd = '';
                $scope.opt.viewIdx = -1;
                renderTask($scope.taskidx);
            }
        };
        $scope.submitCmd = function (command) {
            $rootScope.dataloaded = false;
            if ($scope.submitter) {
                //if($scope.taskidx != 0) $scope.isOpenDir = false;
                $scope.isOpenDir = true;
                //read only - submission already done
                if ($scope.taskidx >= $scope.taskdata.idxlist.length)
                    $scope.taskidx = 0;
                renderTask($scope.taskidx);
            }
            else {
                if ($scope.hitId) {
                    //error check length
                    var myWords = command.replace(/ +/g, ' ').split(' ');
                    if (myWords.length < 3) {
                        toaster.pop('error', 'Not enough words used in description');
                        $rootScope.dataloaded = true;
                        return;
                    }
                    var isValid = true;
                    var cmdlc = command.toLowerCase();
                    var l = Object.keys($scope.cmdlist).length; //get # of keys then length because there are no sparse numbers in associative array num index
                    for (var i = 0; i < l; i++) {
                        if ($scope.cmdlist[i]) {
                            var ldist = levenshtein.get(cmdlc, $scope.cmdlist[i].send.input.toLowerCase());
                            if (ldist < 2) {
                                isValid = false;
                                i = l;
                            }
                        }
                    }
                    if (!isValid) {
                        toaster.pop('error', 'Command is too similiar to previous commands');
                        $rootScope.dataloaded = true;
                        return;
                    }
                    var cmdinput = serialState(command);
                    if (cmdinput) {
                        //submit to AI system simulate wait
                        setTimeout(function () {
                            Meteor.call('cmdMovePost', cmdinput, function (err, ret) {
                                if (err)
                                    return $scope.$apply(function () {
                                        toaster.pop('error', err);
                                    });
                                if (ret.error)
                                    return $scope.$apply(function () {
                                        toaster.pop('error', ret.error);
                                    });
                                if (ret.result) {
                                    var cmdoutput = ret.result;
                                    if (cmdoutput && !cmdoutput.error) {
                                        var states = convCmdToState(cmdinput, cmdoutput);
                                        myengine.updateScene(states, function () {
                                            $scope.cmdele = { send: cmdinput, recv: cmdoutput, rate: -1 };
                                            $scope.$apply(function () {
                                                $rootScope.dataloaded = true;
                                                $scope.cmdphase = eCmdPhase.RATE;
                                            });
                                        });
                                    }
                                    else {
                                        $scope.$apply(function () {
                                            $rootScope.dataloaded = true;
                                            toaster.pop('error', cmdoutput.error);
                                        });
                                    }
                                }
                                else
                                    console.warn(err, ret);
                            });
                        }, 1000);
                    }
                    else
                        $rootScope.dataloaded = true;
                }
                else
                    toaster.pop('error', 'Missing HIT Id');
            }
        };
        $scope.setRating = function (rating) {
            $rootScope.dataloaded = false;
            $scope.cmdele.rate = rating;
            //must use update instead of save because _id is custom generated
            if (rating != 1)
                saveCmd();
            else
                renderFix();
        };
        var renderFix = function () {
            $scope.cmdphase = eCmdPhase.FIX;
            myengine.opt.enableUI = true;
            $scope.resetFix();
            $rootScope.dataloaded = true;
        };
        $scope.resetFix = function () {
            //reset to base view for command move
            $scope.opt.isBaseView = false; //we get toggle in showTransition so we set to the view we DONT want
            $scope.showTransition();
        };
        $scope.submitFix = function () {
            var cmdserial = serialState("fix");
            cmdserial.type = mGenCmdJobs.eCmdType.FIX;
            var fudgeVal = 0.001;
            var deltaidx = [];
            var slen = $scope.cmdele.send.world.length;
            for (var i = 0; i < cmdserial.world.length; i++) {
                var isMatch = false;
                var a = cmdserial.world[i];
                for (var j = 0; j < slen; j++) {
                    var b = $scope.cmdele.send.world[j];
                    if (Math.abs(b.loc[0] - a.loc[0]) < fudgeVal && Math.abs(b.loc[1] - a.loc[1]) < fudgeVal && Math.abs(b.loc[2] - a.loc[2]) < fudgeVal) {
                        isMatch = true;
                        j = slen;
                    }
                }
                if (!isMatch)
                    deltaidx.push(i);
            }
            if (deltaidx.length) {
                $scope.cmdphase = eCmdPhase.CMD;
                myengine.opt.enableUI = false;
                var newWorld = [];
                _.each(deltaidx, function (i) {
                    newWorld.push(cmdserial.world[i]);
                });
                cmdserial.world = newWorld;
                $scope.cmdele.fix = cmdserial;
                saveCmd();
            }
            else
                toaster.pop('info', 'Please move a cube to save a fix.');
        };
        var saveCmd = function () {
            $scope.cmdlist[$scope.taskidx] = $scope.cmdele;
            $scope.cmdele = null;
            var previdx = $scope.taskidx; //get actual index
            $scope.taskidx += 1;
            if (!$scope.hitdata.timed)
                $scope.hitdata.timed = {};
            if (!$scope.hitdata.timed[$scope.workerId])
                $scope.hitdata.timed[$scope.workerId] = {};
            if (!$scope.hitdata.timed[$scope.workerId][previdx])
                $scope.hitdata.timed[$scope.workerId][previdx] = (new Date()).getTime();
            var setdata = {};
            setdata['cmdlist.' + $scope.workerId] = $scope.hitdata.cmdlist[$scope.workerId];
            setdata['timed.' + $scope.workerId] = $scope.hitdata.timed[$scope.workerId];
            GenCmdJobs.update({ _id: $scope.hitdata._id }, {
                $set: setdata
            }, function (err, ret) {
                if (err)
                    return toaster.pop('error', err.reason);
                $scope.$apply(function () {
                    $scope.opt.command = '';
                    $rootScope.dataloaded = true;
                    $scope.cmdphase = eCmdPhase.CMD;
                });
            });
        };
        $scope.submit = function () {
            if ($scope.taskidx >= $scope.maxtask && $scope.assignmentId && $scope.assignmentId != 'ASSIGNMENT_ID_NOT_AVAILABLE') {
                //submission assignment as done
                if (!$scope.hitdata.submitted)
                    $scope.hitdata.submitted = [];
                var subfound = _.findWhere($scope.hitdata.submitted, { name: $scope.workerId });
                if (_.isUndefined(subfound)) {
                    $scope.hitdata.submitted.push({
                        name: $scope.workerId,
                        time: (new Date()).getTime(),
                        aid: $scope.assignmentId,
                        submitto: $scope.turkSubmitTo
                    });
                    $scope.submitter = $scope.hitdata.submitted[$scope.hitdata.submitted.length - 1];
                    $scope.taskidx = 0;
                    GenCmdJobs.update({ _id: $scope.hitdata._id }, {
                        $addToSet: {
                            submitted: $scope.submitter
                        }
                    }, function (err, ret) {
                        //console.warn('hit', err, ret);
                        if (err)
                            return toaster.pop('error', err);
                        $scope.$apply(function () {
                            toaster.pop('info', 'HIT Task Submitted');
                        });
                        $('form[name="submitForm"]').submit(); //submit to turk
                    });
                }
            }
        };
        var serialState = function (cmd) {
            var cmdserial = {
                world: null,
                type: mGenCmdJobs.eCmdType.INPUT,
                input: cmd,
                version: 1
            };
            var newblock_state = [];
            var block_state = $scope.curState.block_state;
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
                return null;
            }
            for (var i = 0; i < frame.length; i++) {
                var s = frame[i];
                var pos = [];
                _.each(s.position, function (v) {
                    pos.push(v);
                });
                newblock_state.push({ id: s.id, loc: pos });
            }
            cmdserial.world = newblock_state;
            return cmdserial;
        };
        $scope.showTransition = function () {
            $scope.opt.isBaseView = !$scope.opt.isBaseView;
            var base = null;
            var delta = null;
            if ($scope.cmdele) {
                base = $scope.cmdele.send;
                if ($scope.cmdele.fix)
                    delta = $scope.cmdele.fix;
                else
                    delta = $scope.cmdele.recv;
            }
            else {
                var cidx = $scope.taskidx - 1; //taskidx shows next empty entry
                if ($scope.opt.viewIdx > -1)
                    cidx = $scope.opt.viewIdx;
                base = $scope.cmdlist[cidx].send;
                if ($scope.cmdlist[cidx].fix)
                    delta = $scope.cmdlist[cidx].fix;
                else
                    delta = $scope.cmdlist[cidx].recv;
            }
            var states = convCmdToState(base, delta);
            myengine.updateScene(states, function () {
                $scope.$apply(function () {
                    $rootScope.dataloaded = true;
                });
            });
        };
        $scope.emailLogin = function (email) {
            if (email.length > 0) {
                var emailenc = $filter('uriCompEnc')(email); //encodeURIComponent(email).replace(/\./g,'%2E'); //encode @ and also . so that mongo doesn't screw things up
                var stateGo = apputils.stateGo($state);
                stateGo('gencmdtask', { taskId: $scope.taskId, assignmentId: $scope.assignmentId, hitId: $scope.hitId, workerId: emailenc });
            }
        };
        $scope.dlScene = function () {
            var content = JSON.stringify($scope.curState, null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_scene_' + $scope.curState._id + '.json');
        };
        $scope.dlStates = function () {
            var content = JSON.stringify($scope.taskdata, null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_states_' + $scope.taskdata._id + '.json');
        };
        $scope.dlNotes = function () {
            var content = JSON.stringify($scope.hitdata, null, 2);
            var uriContent = "data:application/octet-stream," + encodeURIComponent(content);
            apputils.saveAs(uriContent, 'bw_notes_' + $scope.hitdata.HITId + '.json'); //+'_'+$scope.workerId+'.json');
        };
        //todo: remove old functions
        /*$scope.updateReport = function(idx: number, form: angular.IFormController){
          var setdata:{[x: string]:any} = {};
          setdata['cmdlist.'+$scope.workerId] = $scope.hitdata.cmdlist[$scope.workerId];
          GenCmdJobs.update({_id: $scope.hitdata._id}, {
            $set: setdata
          }, function(err, ret){
            if(err) return toaster.pop('error', err.reason);
            form.$setPristine();
          });
        };
      
        $scope.validateReport = function(opt: string){
          var subidx:number = _.findIndex<miGenCmdJobs.iSubmitEle>($scope.hitdata.submitted, function(v:miGenCmdJobs.iSubmitEle){return v.name == $scope.workerId});
          if(subidx>-1) {
            $scope.submitter.valid = opt;
            var setdata:{[x: string]:any} = {};
            setdata['submitted.'+subidx] = $scope.submitter;
            GenCmdJobs.update({_id: $scope.hitdata._id}, {
              $set: setdata
            }, function(err, ret){
              if(err) return toaster.pop('error', err.reason);
            });
          }
        };*/
        var levenshtein = $window.Levenshtein;
        // Start by calling the createScene function that you just finished creating
        var myengine = new mGen3DEngine.cUI3DEngine(APP_CONST.fieldsize);
        myengine.opt.enableUI = false;
        myengine.createWorld();
        dataReady.update('world created');
    }]);
//# sourceMappingURL=gen-cmdtask-view.js.map