/**========================================================
 * Module: gen-cmdexp-view
 * Created by wjwong on 1/26/16.
 =========================================================*/
/// <reference path="gen-3d-engine.ts" />
/// <reference path="../../../../../model/gencmdsdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />
/// <reference path="../../../../../server/cmdmoveshelper.ts" />

angular.module('app.generate').controller('genCmdExpCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', 'toaster', 'APP_CONST', 'DTOptionsBuilder', 'AppUtils', '$reactive', function($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, toaster, APP_CONST, DTOptionsBuilder, apputils, $reactive) {
  "use strict";
  $reactive(this).attach($scope);

  //if user is not logged and during this view and we log in then fire a reload
  Accounts.onLogin(function (user) {$state.reload();});

  $scope.isGuest = $rootScope.isRole(Meteor.user(), 'guest');
  //subscription error for onStop;
  var subErr:(err:Error)=>void = function(err:Error){if(err) console.warn("err:", arguments, err); return;};

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
    "language": {"paginate": {"next": '▶', "previous": '◀'}},
    "dom": '<"pull-left"f><"pull-right"i>rt<"pull-left"p>'
  });

  if($scope.isGuest)
    $scope.dtOptionsAvail.order = [[2, "desc"]];
  else
    $scope.dtOptionsAvail.order = [[3, "desc"]];

  $scope.curState = new apputils.cCurrentState();

  setTimeout(()=>{
    $scope.subscribe("gencmds", ()=>{}, {
      onReady: function (sid) {dataReady.update('gencmds')},
      onStop: subErr
    });
  }, 10);

  var dataReady:iDataReady = new apputils.cDataReady(1, function ():void {
    updateAvailExp();
    if ($stateParams.sid) {
      $scope.showState($stateParams.sid);
    }
    else $rootScope.dataloaded = true;
  });

  var updateAvailExp = function () {
    $scope.availExp = <iGenCmds[]>GenCmds.find({}, {sort: {"_id": 1}}).fetch();
  };

  $scope.resetWorld = function () {
    //resetworld 
    myengine.resetWorld();
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
  };

  /*var findBy = function(type:string, key:string, collection:any){
   return _.find(collection, function(a){return key === a[type]});
   };*/

  var showImage = function (b64i:string, text:string, attachID:string) {
    var b64img:string = LZString.decompressFromUTF16(b64i);

    var eleDivID:string = 'div' + $('div').length; // Unique ID
    var eleImgID:string = 'img' + $('img').length; // Unique ID
    //var eleLabelID:string = 'h4' + $('h4').length; // Unique ID
    var htmlout:string = '';
    if (text) htmlout += '<b>' + text + '</b><br>';
    htmlout += '<img id="' + eleImgID + '" style="width:' + myengine.canvas.width * 2 / 3 + 'px;height:' + myengine.canvas.height * 2 / 3 + 'px"></img>';
    // + '<label id="'+eleLabelID+'" class="mb"> '+id+'</label>';
    var attachTo = '#galleryarea';
    if (attachID) attachTo = '#' + attachID;
    $('<div>').attr({
      id: eleDivID
    }).addClass('col-sm-12')
      .html(htmlout).css({}).appendTo(attachTo);

    var img:HTMLImageElement = <HTMLImageElement>document.getElementById(eleImgID); // Use the created element
    img.src = b64img;
  };

  var checkFnSS:number; //store steady state check

  /**
   * show the state to be used as state 0
   * @param sid
   */
  $scope.showState = function (sid:string) {
    if(!$stateParams.sid) //fix double routing when theres an update
      $state.transitionTo('app.gencmdexp', {sid: sid}, {notify: false});
    $rootScope.dataloaded = false;
    $scope.enableImpSave = false;
    $scope.isExp = true;
    //we must get the state for this sid
    $scope.subscribe("gencmds", ()=>{return [sid]}, {
      onReady: function (sub) {
        var myframe:iGenCmds = GenCmds.findOne({_id: sid});
        if (!myframe){
          $rootScope.dataloaded = true;
          toaster.pop('warn', 'Invalid State ID');
          $state.transitionTo('app.gencmdexp', {}, {notify: false});
          return;
        }
        //update the meta
        $scope.curState.clear();
        $scope.curState.copy(myframe);
        myengine.createObjects($scope.curState.block_meta.blocks);
        myengine.updateScene({block_state: myframe.block_state});
        $scope.$apply(()=>{$rootScope.dataloaded = true});
      },
      onStop: subErr
    })
  };

  $scope.remState = function (sid:string) {
    if (sid){
      GenCmds.remove(sid);
      updateAvailExp();
      toaster.pop('warning', 'Removed ' + sid);
    }
  };

  var getStackCubes = function (mycube:miGen3DEngine.iCubeState, used:miGen3DEngine.iCubeState[], cid:number, checkY:boolean):miGen3DEngine.iCubeState[] {
    var retStack:miGen3DEngine.iCubeState[] = [];
    for (var i = 0; i < used.length; i++) {
      if (!cid || cid != used[i].prop.cid) {
        var c = used[i];
        if (myengine.intersectsMeshXYZ(mycube, c, checkY)) {
          retStack.push(c);
        }
      }
      //else console.warn('skipped', cid)
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

  $scope.saveImport = function (savename:string, isMulti?:boolean, cb?:()=>void) {
    $rootScope.dataloaded = false;

    $scope.impFilename = null;
    $scope.enableImpSave = false;
    $scope.curState.name = savename;
    setTimeout(function () {
      GenCmds.insert(angular.copy($scope.curState), function(err: Error, id:string) {
        if(err){
          $rootScope.dataloaded = true;
          $scope.clearMeta();
          $scope.$apply(()=>{toaster.pop('error', 'Save Import error: ', err.message)});
          return;
        }
        if(!isMulti){
          $scope.curState._id = id;
          $rootScope.dataloaded = true;
          $state.go('app.gencmdexp', {sid: id}, {reload:true, notify: true});
        }
        if(cb) cb();
        //$state.transitionTo('app.gencmdexp', {sid: val[0]._id}, {notify: false});
      });
    }, 400);
  };

  $scope.clearMeta = function () {
    $('#galleryarea').empty();
    $scope.curState.clear();
    $state.transitionTo('app.gencmdexp', {}, {notify: false});
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
      })
    }
  }

  $scope.loadStates = function () {
    if ($scope.statesfilename && $scope.statesfilename.length) {
      if ($scope.statesfilename.length > 1) {//multi file
        var reader = new FileReader();

        var procFiles = function(idx:number, files:Blob[], cb:()=>void) {
          if(_.isUndefined(files[idx])) return cb();
          reader.onload = function () {
            var filedata:miGen3DEngine.iBlockImport = JSON.parse(reader.result);
            var validKeys:string[] = ['block_meta', 'block_state', 'name'];
            var resValidKeys:iRetValue = apputils.isValidKeys(filedata, validKeys);
            if (resValidKeys.ret && filedata.block_state.length
              && filedata.block_meta.blocks && filedata.block_meta.blocks.length
            ) {
              if (filedata.block_meta.blocks.length != filedata.block_state.length) return $scope.$apply(function () {
                toaster.pop('error', 'Block META and STATE mismatch! '+files[idx]);
                procFiles(idx+1, files, cb);
              });
              $scope.curState.clear();
              $scope.curState.block_meta = filedata.block_meta;
              $scope.curState.public = true;
              $scope.curState.created = (new Date).getTime();
              $scope.curState.creator = $rootScope.currentUser._id;
              $scope.curState.type = 'sc'; //scene type
              setDecorVal(filedata.block_meta.decoration);
              $scope.curState.block_state = mungeBlockState(filedata.block_state);
              var savename:string = files[idx]['name'].toLowerCase().replace(/\.json/g, '');
              $scope.saveImport(savename, true, function(){
                toaster.pop('info','Saved '+savename);
                procFiles(idx+1, files, cb);
              });
            }
            else $scope.$apply(function () {
              toaster.pop('warn', 'Invalid JSON '+files[idx], JSON.stringify(resValidKeys.err));
              procFiles(idx+1, files, cb);
            });
          };
          reader.readAsText(files[idx]);
        };

        procFiles(0, $scope.statesfilename, function(){
          $rootScope.dataloaded = true;
          $scope.curState.clear();
          updateAvailExp();
          $state.go('app.gencmdexp', {}, {reload:true, notify: true});
        })
      }
      else {
        $scope.isExp = false;
        //read file
        var reader = new FileReader();
        reader.onload = function () {
          var filedata:miGen3DEngine.iBlockImport = JSON.parse(reader.result);
          var validKeys:string[] = ['block_meta', 'block_state', 'name'];
          var resValidKeys:iRetValue = apputils.isValidKeys(filedata, validKeys);
          if (resValidKeys.ret && filedata.block_state.length
            && filedata.block_meta.blocks && filedata.block_meta.blocks.length
          ) {
            if (filedata.block_meta.blocks.length != filedata.block_state.length) return $scope.$apply(function () {
              toaster.pop('error', 'Block META and STATE mismatch!');
            });
            $scope.curState.clear();
            $scope.curState.block_meta = filedata.block_meta;
            $scope.curState.public = true;
            $scope.curState.created = (new Date).getTime();
            $scope.curState.creator = $rootScope.currentUser._id;
            $scope.curState.type = 'sc'; //scene type
            setDecorVal(filedata.block_meta.decoration);
            myengine.createObjects($scope.curState.block_meta.blocks);
            //mung block_states
            $scope.curState.block_state = mungeBlockState(filedata.block_state);
            $scope.$apply(function () {
              $scope.impFilename = null;
              $scope.enableImpSave = false;
              $scope.isgen = true;
            });

            myengine.updateScene({block_state: $scope.curState.block_state}, function () {
              //wait for steady state
              checkFnSS = setInterval(function () {
                if (myengine.isSteadyState) {
                  clearInterval(checkFnSS);
                  $scope.$apply(function () {
                    if (filedata.name) $scope.impFilename = filedata.name;
                    else $scope.impFilename = $scope.statesfilename[0].name.toLowerCase().replace(/\.json/g, '');
                    $scope.enableImpSave = true;
                    $scope.isgen = false;
                  });
                }
              }, 100);
            });
          }
          else $scope.$apply(function () {
            toaster.pop('warn', 'Invalid JSON STATE file', JSON.stringify(resValidKeys.err))
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

  var mungeBlockStates = function (bss:miGen3DEngine.iBlockStatesSerial[]):iBlockStates[] {
    var newbss:iBlockStates[] = [];
    for (var i = 0; i < bss.length; i++) {
      newbss.push({block_state: mungeBlockState(bss[i].block_state)});
    }
    return newbss;
  };


  /**
   * Transform text block state from cwic to internal block states
   * @param bs
   * @returns {Array}
   */
  var mungeBlockState = function (bs:miGen3DEngine.iBlockStateSerial[]):iBlockState[] {
    var newBS:iBlockState[] = [];
    bs.forEach(function (b) {
      if(typeof b.position != "string") $scope.$apply(()=>{toaster.pop('error', "Invalid: position should be a string list of coordinates")});
      var li:string[] = b.position.split(',');
      var lv:number[] = [];
      li.forEach(function (v, i) {
        lv.push(Number(v))
      });
      if (b.rotation) {
        var ri:string[] = b.rotation.split(',');
        var rv:number[] = [];
        ri.forEach(function (v, i) {
          rv.push(Number(v))
        });
        newBS.push({
          id: b.id, position: {
            x: lv[0], y: lv[1], z: lv[2]
          }, rotation: {
            x: rv[0], y: rv[1], z: rv[2], w: rv[3]
          }
        })
      }
      else
        newBS.push({
          id: b.id, position: {
            x: lv[0], y: lv[1], z: lv[2]
          }
        })

    });
    return newBS;
  };
  
  // Start by calling the createScene function that you just finished creating
  var myengine:miGen3DEngine.c3DEngine = new mGen3DEngine.c3DEngine(APP_CONST.fieldsize);

  $scope.opt = myengine.opt;
  $scope.opt.limStack = true; //we add a stack limit to 3d engine vars
  $scope.isExp = true; //all work is consider experiment view unless we import a state
  myengine.createWorld();
  dataReady.update('world created');
}]);