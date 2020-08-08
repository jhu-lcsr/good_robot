/**========================================================
 * Module: gen-task-view.ts
 * Created by wjwong on 10/5/15.
 =========================================================*/
/// <reference path="../../../../../model/genjobsmgrdb.ts" />
/// <reference path="../../../../../model/genstatesdb.ts" />
/// <reference path="../../../../../model/screencapdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/jquery/jquery.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../services/apputils.ts" />

//http://localhost:3000/annotate?taskId=hDhCMtviYMx9CZmpn&assignmentId=test&workerId=wjwong&hitId=3PA41K45VNQJK7JZ308ZJRKNJEM7PV&turkSubmitTo=https://www.mturk.com/

interface iSceneInfo {
  _id: string, public: string, name: string, created: string,
  creator: string, block_meta: iBlockMeta, block_states: iBlockStates[]
}

angular.module('app.generate').controller('genTaskCtrl', ['$rootScope', '$scope', '$state', '$stateParams', '$translate', '$window', '$localStorage', '$timeout', '$reactive', 'ngDialog', 'toaster', 'AppUtils', 'deviceDetector', function($rootScope, $scope, $state, $stateParams, $translate, $window, $localStorage, $timeout, $reactive, ngDialog, toaster, apputils, devDetect){
  "use strict";
  $reactive(this).attach($scope);
  //subscription error for onStop;
  var subErr:(err:Error)=>void = function(err:Error){if(err) console.warn("err:", arguments, err); return;};

  $scope.date = (new Date()).getTime();
  $scope.opt = {bAgreed: true, repvalidlist: [mGenJobsMgr.eRepValid[0], mGenJobsMgr.eRepValid[1], mGenJobsMgr.eRepValid[2]], repvalid: '', isValidBrowser: (devDetect.browser.toLowerCase() === 'chrome')};
  
  $scope.subscribe("genstates", ()=>{}, {
    onReady: function (sid) {
      dataReady.update('genstates');
    },
    onStop: subErr
  });
  $scope.subscribe("screencaps", ()=>{}, {
    onReady: function (sid) {
      dataReady.update('screencaps');
    },
    onStop: subErr
  });
  $scope.subscribe("genjobsmgr", ()=>{}, {
    onReady: function (sid) {
      dataReady.update('genjobsmgr');
    },
    onStop: subErr
  });

  $scope.isOpenDir = true;
  $scope.taskdata;
  $scope.taskidx = -1;
  $scope.maxtask = -1;
  $scope.curantpass = -1;
  $scope.notes = null;
  var dataReady:iDataReady = new apputils.cDataReady(2, function():void {
    var isAdminUser = ($rootScope.currentUser) ? $rootScope.isRole($rootScope.currentUser, 'admin') : false;
    if ($stateParams.report && !isAdminUser) { //not admin we just leave it blank
      $rootScope.dataloaded = true;
      return;
    }
    if ($stateParams.taskId){ //get task information and start loading
      $scope.taskdata = <miGenJobsMgr.iGenJobsMgr>GenJobsMgr.findOne($stateParams.taskId);
      if (!$scope.taskdata) {
        $rootScope.dataloaded = true;
        $scope.assignmentId = null;
        return;
      }
      $scope = _.extend($scope, $stateParams);
      if ($scope.turkSubmitTo) $scope.submitTo = $scope.turkSubmitTo + '/mturk/externalSubmit';
      if ($scope.workerId === 'EXAMPLE') $scope.submitter = true;
      if(!$scope.assignmentId && !$stateParams.report && !$stateParams.json){
        $rootScope.dataloaded = true;
        return;
      }

      if ($scope.hitId) {
        //load hit
        $scope.hitdata = <miGenJobsMgr.iGenJobsHIT>GenJobsMgr.findOne('H_' + $scope.hitId);
        if ($scope.hitdata && $scope.hitdata.submitted && $scope.workerId && $scope.workerId !== 'EXAMPLE') {
          var subfound:miGenJobsMgr.iSubmitEle = <miGenJobsMgr.iSubmitEle>_.findWhere($scope.hitdata.submitted, {name: $scope.workerId});
          if (!_.isUndefined(subfound)) {//check if its already submitted by this worker
            //worker already submitted
            $scope.submitter = subfound;
          }
        }
      }
      var sid:string = $scope.taskdata.stateid;
      $scope.subscribe("genstates", ()=>{return [sid]}, {
        onReady: function (sub) {
          $scope.curState = <iGenStates>GenStates.findOne(sid);
          //console.warn('curState',$scope.curState);
          if ($stateParams.report) { //report view
            $scope.report = $stateParams.report;
            if ($scope.submitter.valid)
              $scope.opt.repvalid = $scope.submitter.valid;
            else $scope.opt.repvalid = 'tbd';
            if ($scope.hitdata.notes[$scope.workerId]) {
              $rootScope.dataloaded = true;  //turn off loading so one can quickly get data.
              $timeout(function () {
                renderReport(0)
              });
            }
            else {
              $rootScope.dataloaded = true;
              toaster.pop('error', 'Missing annotations');
            }
          }
          else {//single item view
            $scope.maxtask = $scope.taskdata.idxlist.length * $scope.taskdata.antcnt;
            $scope.taskidx = 0;
            if ($scope.submitter) {
              $scope.curantpass = $scope.taskdata.antcnt;
            }
            else {
              if ($scope.hitdata && $scope.hitdata.notes && $scope.hitdata.notes[$scope.workerId]) {
                //we have hit data lets fast forward to the correct item to work on
                //assume that we fill notes from pass 1 then pass 2 etc. there are no holes in the list
                var mynotes:string[][] = $scope.hitdata.notes[$scope.workerId];
                _.each(mynotes, function (n) {
                  n.forEach(function (i) {
                    if (i.length) $scope.taskidx++;
                  })
                })
              }
              $scope.curantpass = Math.floor($scope.taskidx / $scope.taskdata.idxlist.length);
            }
            if ($scope.taskidx || $scope.submitter) $scope.opt.bAgreed = true;
            renderTask($scope.taskidx);
            $scope.logolist = [];
            _.each($scope.curState.block_meta.blocks, function (b:iBlockMetaEle) {
              $scope.logolist.push({name: b.name, imgref: "img/textures/logos/" + b.name.replace(/ /g, '') + '.png'});
            });
          }
          /*Meteor.call('mturkReviewableHITs', {hid: $scope.hitId},  function(err, resp){
           console.warn(err,resp);
           })*/
        },
        onStop: subErr
      });
    }
  });

  var renderReport = function(idx:number){
    if(_.isUndefined($scope.hitdata.notes[$scope.workerId][idx])){ //stop at where the worker notes stop
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
  };
  
  var renderTask = function(tidx:number){
    if($scope.taskidx != 0) $scope.isOpenDir = false;
    else $scope.isOpenDir = true;
    //convert to actual index
    var idx:number = tidx%$scope.taskdata.idxlist.length;
    //create the annotations
    if($scope.hitdata){
      if(!$scope.hitdata.notes) $scope.hitdata.notes = {};
      if(!$scope.hitdata.notes[$scope.workerId]) $scope.hitdata.notes[$scope.workerId] = {};
      if(!$scope.hitdata.notes[$scope.workerId][idx]){
        $scope.hitdata.notes[$scope.workerId][idx] = [];
        for(var i =0; i < $scope.taskdata.antcnt; i++)
          $scope.hitdata.notes[$scope.workerId][idx].push('');
      }
      $scope.notes = $scope.hitdata.notes[$scope.workerId][idx];
    }
    else{//only an example no HIT id - only need to show 1 note
      $scope.notes = [];
      $scope.notes.push('');
    }
    if($scope.taskdata.tasktype == 'action'){
      var aidx:number = $scope.taskdata.idxlist[idx][0];
      var bidx:number = $scope.taskdata.idxlist[idx][1];
      $('#statea').empty();
      $('#stateb').empty();
      var scids:string[] = [$scope.curState.block_states[aidx].screencapid, $scope.curState.block_states[bidx].screencapid];
      $scope.subscribe('screencaps', ()=>{return [scids]}, {
        onReady: function(sub){
          var screena:iScreenCaps = ScreenCaps.findOne(scids[0]);
          var screenb:iScreenCaps = ScreenCaps.findOne(scids[1]);
          showImage(screena.data, 'Before', null, 'statea');
          showImage(screenb.data, 'After', null, 'stateb');
          $rootScope.dataloaded = true;
        },
        onStop: subErr
      });
    }
  };
  
  var showImage = function(b64i:string, title:string, caption:string, attachID:string){
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
  
  $scope.itrAnnot = function(notes:string[], vdir:number){
    $rootScope.dataloaded = false;
    if($scope.submitter){
      $scope.taskidx+=vdir;
      if($scope.taskidx != 0) $scope.isOpenDir = false;
      else $scope.isOpenDir = true;
      //read only - submission already done
      if($scope.taskidx >= $scope.taskdata.idxlist.length) $scope.taskidx = 0;
      $scope.curantpass = $scope.taskdata.asncnt;
      renderTask($scope.taskidx);
    }
    else{//new entry save as we go
      if($scope.hitId){
        if(vdir<0){
          $scope.taskidx+=vdir;
          $scope.curantpass = Math.floor($scope.taskidx/$scope.taskdata.idxlist.length);
          renderTask($scope.taskidx);
          return;
        }
        //error check length
        var fixedNotes:string[] = _.compact(notes);
        console.warn(fixedNotes, $scope.curantpass);
        //since we are filling in annotates 1 at a time over N passes - we just need to check length based on currant whole pass + 1
        if(!fixedNotes.length || fixedNotes.length != $scope.curantpass+1 ){
          toaster.pop('error', 'All entries must be filled.');
          $rootScope.dataloaded = true;
          return;
        }
        //check uniq entries
        fixedNotes = _.uniq(fixedNotes);
        if(!fixedNotes.length || fixedNotes.length != $scope.curantpass+1 ){
          toaster.pop('error', 'Each description must be different.');
          $rootScope.dataloaded = true;
          return;
        }
        //check for number of words
        /*var validWords:boolean = true;
        _.each(notes, function(n){
          if(n.split(' ').length < 4) validWords = false;
        });
        if(!validWords){*/
        var myWords:string[] = notes[$scope.curantpass].replace(/ +/g,' ').split(' ');
        if((!$scope.curState.type && myWords.length < 4) ||
          ($scope.curState.type && myWords.length < 20)){
          toaster.pop('error', 'Not enough words used in description');
          $rootScope.dataloaded = true;
          return;
        }

        var previdx:number = ($scope.taskidx)%$scope.taskdata.idxlist.length; //get actual index
        $scope.taskidx+=vdir;
        $scope.curantpass = Math.floor($scope.taskidx/$scope.taskdata.idxlist.length);

        if(!$scope.hitdata.timed) $scope.hitdata.timed = {};
        if(!$scope.hitdata.timed[$scope.workerId]) $scope.hitdata.timed[$scope.workerId] = {};
        if(!$scope.hitdata.timed[$scope.workerId][previdx]) $scope.hitdata.timed[$scope.workerId][previdx] = (new Date()).getTime();

        //must use update instead of save because _id is custom generated
        var setdata:{[x: string]:any} = {};
        setdata['notes.'+$scope.workerId] = $scope.hitdata.notes[$scope.workerId];
        setdata['timed.'+$scope.workerId] = $scope.hitdata.timed[$scope.workerId];
        GenJobsMgr.update({_id: $scope.hitdata._id}, {
          $set: setdata
        }, function(err, ret){
          if(err) return toaster.pop('error', err.reason);
          if($scope.taskidx >= $scope.maxtask && $scope.assignmentId && $scope.assignmentId != 'ASSIGNMENT_ID_NOT_AVAILABLE'){
            //submission assignment as done
            if(!$scope.hitdata.submitted) $scope.hitdata.submitted = [];
            var subfound:miGenJobsMgr.iSubmitEle = <miGenJobsMgr.iSubmitEle>_.findWhere($scope.hitdata.submitted, {name: $scope.workerId});
            if(_.isUndefined(subfound)){
              $scope.hitdata.submitted.push({
                name: $scope.workerId,
                time: (new Date()).getTime(),
                aid: $scope.assignmentId,
                submitto: $scope.turkSubmitTo
              });
              $scope.submitter = $scope.hitdata.submitted[$scope.hitdata.submitted.length-1];
              $scope.taskidx = 0;
              $scope.curantpass = 0;
              GenJobsMgr.update({_id: $scope.hitdata._id}, {
                $addToSet: {
                  submitted: $scope.submitter
                }
              }, function(err, ret){
                console.warn('hit', err, ret);
                if(err) return toaster.pop('error', err);
                $scope.$apply(function(){toaster.pop('info', 'HIT Task Submitted')});
                $('form[name="submitForm"]').submit(); //submit to turk
              });
            }
          }
          else renderTask($scope.taskidx);
        });
      }
      else toaster.pop('error', 'Missing HIT Id');
    }
  };

  $scope.updateReport = function(idx: number, form: angular.IFormController){
    var setdata:{[x: string]:any} = {};
    setdata['notes.'+$scope.workerId] = $scope.hitdata.notes[$scope.workerId];
    GenJobsMgr.update({_id: $scope.hitdata._id}, {
      $set: setdata
    }, function(err, ret){
      if(err) return toaster.pop('error', err.reason);
      form.$setPristine();
    });
  };
  
  $scope.validateReport = function(opt: string){
    var subidx:number = _.findIndex<miGenJobsMgr.iSubmitEle>($scope.hitdata.submitted, function(v:miGenJobsMgr.iSubmitEle){return v.name == $scope.workerId});
    if(subidx>-1) {
      $scope.submitter.valid = opt;
      var setdata:{[x: string]:any} = {};
      setdata['submitted.'+subidx] = $scope.submitter;
      GenJobsMgr.update({_id: $scope.hitdata._id}, {
        $set: setdata
      }, function(err, ret){
        if(err) return toaster.pop('error', err.reason);
      });
    }
  };
  
  /*var compileScene = function():iSceneInfo{
    var tempframe:iSceneInfo = {_id: $scope.curState._id,
      public: $scope.curState.public, name: $scope.curState.name, created: $scope.curState.created,
      creator: $scope.curState.creator, block_meta: $scope.curState.block_meta, block_states: []};

    for(var idx = 0; idx < $scope.curState.block_states.length; idx++){
      var block_state = $scope.curState.block_states[idx].block_state;
      var newblock_state = [];
      for(var i = 0; i < block_state.length; i++){
        var s = block_state[i];
        var pos = '', rot = '';
        _.each(s.position, function(v){
          if(pos.length) pos += ',';
          pos += v;
        });
        _.each(s.rotation, function(v){
          if(rot.length) rot += ',';
          rot += v;
        });
        newblock_state.push({id: s.id, position: pos, rotation: rot})
      }
      tempframe.block_states.push({block_state: newblock_state});
    }
    return tempframe;
  };*/
  
  $scope.dlScene = function(){
    var content:string = JSON.stringify($scope.curState, null, 2);
    var uriContent:string = "data:application/octet-stream," + encodeURIComponent(content);
    apputils.saveAs(uriContent, 'bw_scene_'+$scope.curState._id+'.json');
  };

  $scope.dlStates = function(){
    var content:string = JSON.stringify($scope.taskdata, null, 2);
    var uriContent:string = "data:application/octet-stream," + encodeURIComponent(content);
    apputils.saveAs(uriContent, 'bw_states_'+$scope.taskdata._id+'.json');
  };

  $scope.dlNotes = function(){
    var content:string = JSON.stringify($scope.hitdata, null, 2);
    var uriContent:string = "data:application/octet-stream," + encodeURIComponent(content);
    apputils.saveAs(uriContent, 'bw_notes_'+$scope.hitdata.HITId+'.json'); //+'_'+$scope.workerId+'.json');
  };
  
}]);
