/**========================================================
 * Module: gen-jobs-view.ts
 * Created by wjwong on 9/23/15.
 =========================================================*/
/// <reference path="../../../../../model/genjobsmgrdb.ts" />
/// <reference path="../../../../../model/genstatesdb.ts" />
/// <reference path="../../../../../model/screencapdb.ts" />
/// <reference path="../../../../../server/typings/lodash/lodash.d.ts" />
/// <reference path="../../../../../server/typings/lz-string/lz-string.d.ts" />
/// <reference path="../../../../../server/typings/meteor/meteor.d.ts" />
/// <reference path="../../../../../server/typings/jquery/jquery.d.ts" />
/// <reference path="../../../../../server/typings/angularjs/angular.d.ts" />
/// <reference path="../../../../../server/mturkhelper.ts" />
/// <reference path="../services/apputils.ts" />

interface iRepValid {
  [index:string]: string
}

interface iSortHITs {
  time: number,
  name?: string,
  names?: string[],
  repvalid?: iRepValid,
  tid: string, hid: string, islive: boolean,
  reward: {
    Amount: number,
    CurrencyCode: string
  }
}

interface iSortASNs {
  time: number, name: string, tid: string, hid: string, islive: boolean
}

angular.module('app.generate').controller('genJobsCtrl', ['$rootScope', '$scope', '$state', '$translate', '$window', '$localStorage', '$timeout',  'ngDialog', 'toaster', 'AppUtils', 'DTOptionsBuilder', '$reactive', function($rootScope, $scope, $state, $translate, $window, $localStorage, $timeout, ngDialog, toaster, apputils, DTOptionsBuilder, $reactive){
  "use strict";
  $reactive(this).attach($scope);

  $scope.opt= {}; //angular has issues with updating primitives
  $scope.opt.isLive = false;
  $scope.opt.useQual = true;
  $scope.opt.pageCur = 0;
  $scope.opt.pageSize = 100;

  var canvas = {width: 480, height: 360};
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
   "order": [[3, "desc"]],
   "language": {"paginate": {"next": '▶', "previous": '◀'}},
   "dom": '<"pull-left"f><"pull-right"i>rt<"pull-left"p>'
   });

  $scope.dtOptionsGrp = _.extend({}, $scope.dtOptionsAvail, {
    "lengthMenu": [[10], [10]],
    "order": [[2, "desc"]],
  });

  $scope.dtOptionsTask = _.extend({}, $scope.dtOptionsAvail, {
    "lengthMenu": [[10], [10]],
    "order": [[0, "desc"]],
  });
  
  setTimeout(()=>{
    $scope.subscribe("genstates", ()=>{}, {
      onReady: function (sid) {dataReady.update('genstates');},
      onStop: subErr
    });

    $scope.subscribe("screencaps", ()=>{}, {
      onReady: function (sid) {dataReady.update('screencaps');},
      onStop: subErr
    });

    $scope.subscribe("genjobsmgr", ()=>{return [{type: "list", pageCur: $scope.opt.pageCur, pageSize: $scope.opt.pageSize}]}, {
      onReady: function (sid) {dataReady.update('genjobsmgr');},
      onStop: subErr
    });
  }, 10);

  var dataReady:iDataReady = new apputils.cDataReady(2, function():void{
    updateTableStateParams();
    updateJobMgr();
    $scope.refreshHITs();
    $scope.$apply(()=>{$rootScope.dataloaded = true;});
  });

  /*var updateHITs = function(){
    //$scope.doneASNs = getDoneASNs();
    $scope.allHITs = getAllHITs();
  };*/
  
  $scope.incBlock = function(dir:number){
    if($scope.opt.pageCur+dir > -1 ){
      $scope.subscribe("genjobsmgr", ()=>{return [{type: "list", pageCur: $scope.opt.pageCur+dir, pageSize: $scope.opt.pageSize}]}, {
        onReady: function (sid) {
          $scope.opt.pageCur+= dir;
          updateJobMgr();
          $scope.refreshHITs();
        },
        onStop: subErr
      });
    }
  };
  
  $scope.refreshHITs = function(){
    $scope.goodHITsData = true;
    $scope.allHITs = getAllHITs();
    //updateHITs();
    toaster.pop('info', 'Refreshing HITs');
  };

  var getDoneASNs = function(): iSortHITs[]{
    var jobs:miGenJobsMgr.iGenJobsHIT[] = GenJobsMgr.find(
      {$and: [{HITId: {$exists: true}}, {submitted: {$exists: true}}]}
      , {fields: {tid: 1, 'submitted.name': 1, 'submitted.time': 1, 'islive': 1}, sort: {'submitted.time': -1}}
    ).fetch();
    var sortedjobs = [];
    _.each(jobs, function(j){
      j.submitted.forEach(function(h){
        sortedjobs.push({time: h.time, name: h.name, tid: j.tid, hid: j._id.split('_')[1], islive: j.islive})
      })
    });
    if(sortedjobs.length)
      return sortedjobs.sort(function(a:{time: number},b:{time: number}):number{return a.time - b.time});
    return null;
  };

  var getAllHITs= function(): {active: iSortHITs[], done: iSortHITs[], doneASNs: iSortASNs[]}{
    var jobs:miGenJobsMgr.iGenJobsHIT[] = GenJobsMgr.find(
      {HITId: {$exists: true}}
      , {fields: {tid: 1, jid: 1, 'submitted.name': 1, 'submitted.valid': 1, 'submitted.time': 1, 'hitcontent.MaxAssignments': 1, 'hitcontent.Reward': 1, 'created': 1, 'islive': 1}}
      , {sort: {'created': -1}}
    ).fetch();
    var activeHITs = [];
    var doneHITs = [];
    var sortedjobs = [];
    _.each(jobs, function(j:miGenJobsMgr.iGenJobsHIT){
      //hack to store state for the job so we can search easier
      var myjob:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: j.jid});
      if(myjob) j['sid'] = myjob.stateid;

      var asnleft = (j.hitcontent) ? (j.submitted) ? j.hitcontent.MaxAssignments - j.submitted.length : j.hitcontent.MaxAssignments : -1;
      var names = null, repvalid:iRepValid = null;
      if(j.submitted){
        names = [];
        repvalid = {};
        j.submitted.forEach(function(h){
          names.push(h.name);
          if(h.valid) repvalid[h.name] = h.valid;
          else repvalid[h.name] = mGenJobsMgr.eRepValid[mGenJobsMgr.eRepValid.tbd];
          sortedjobs.push({time: h.time, name: h.name, tid: j.tid, hid: j._id.split('_')[1], islive: j.islive})
        })
      }
      if(asnleft > 0)
        activeHITs.push({time: j.created, names: names, tid: j.tid, jid: j.jid, sid: j['sid'], hid: j._id.split('_')[1], asnleft: asnleft, islive: j.islive, reward: j.hitcontent.Reward});
      else {
        var submitTime:number = 0;
        _.each(j.submitted, function(s:miGenJobsMgr.iSubmitEle){
          var t:number = Number(s.time);
          if(t > submitTime) submitTime = t;
        });
        doneHITs.push({time: submitTime, names: names, repvalid: repvalid, tid: j.tid, jid: j.jid, sid: j['sid'], hid: j._id.split('_')[1], asnleft: asnleft, islive: j.islive, reward: j.hitcontent.Reward});
      }
    });

    if(activeHITs.length || doneHITs.length || sortedjobs.length) {
      if (activeHITs.length)
        activeHITs.sort(function (a:{time: number}, b:{time: number}):number {
          return a.time - b.time
        });
      if (doneHITs.length)
        doneHITs.sort(function (a:{time: number}, b:{time: number}):number {
          return a.time - b.time
        });
      if(sortedjobs.length)
        sortedjobs.sort(function(a:{time: number},b:{time: number}):number{return a.time - b.time});
      return {active: activeHITs, done: doneHITs, doneASNs: sortedjobs}
    }
    return null;
  };

  $scope.dlLinks = function(task:iSortHITs, onlyValid:boolean){
    var mytask:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: task.tid});
    var mystate:iGenStates = GenStates.findOne({_id: mytask.stateid});

    var content:string[] = [];
    var htmlcontent:{ex: string, res: string[], st: string} = {ex: '', res: [], st: ''};
    var href:string = '';
    content.push('HIT: '+task.hid);
    content.push('State: '+mystate._id+'  Name: '+mystate.name);
    content.push('Example:');
    href = $state.href('gentask',{taskId: task.tid, assignmentId: 'ASSIGNMENT_ID_NOT_AVAILABLE', workerId: 'EXAMPLE'}, {absolute: true});
    content.push(href);
    htmlcontent.ex = href;
    content.push('Results:');
    _.each(task.names, function(n){
      if(!onlyValid || task.repvalid[n] === mGenJobsMgr.eRepValid[mGenJobsMgr.eRepValid.yes] ){
        href = $state.href('gentask',{taskId: task.tid, workerId: n, hitId: task.hid, report: 1}, {absolute: true});
        content.push(href);
        htmlcontent.res.push(href);
      }
    });

    var uriContent:string = "data:application/octet-stream," + encodeURIComponent(content.join('\n'));
    var fname = 'bw_links_'+task.tid+((onlyValid)? '_v':'')+'.txt';
    apputils.saveAs(uriContent, fname);


    htmlcontent.st = $state.href('app.genworld',{sid: mytask.stateid}, {absolute: true});
    var htmldata = "<body>";
    htmldata += "<h2>HIT: "+task.hid+"</h2>";
    htmldata += "<h4>State</h4>";
    htmldata += "<a href='"+htmlcontent.st+"' target='_blank'>"+htmlcontent.st+"</a><br>";
    htmldata += "<h4>Name: "+mystate.name+"</h4>";
    htmldata += "<h4>Example</h4>";
    htmldata += "<a href='"+htmlcontent.ex+"' target='_blank'>"+htmlcontent.ex+"</a><br>";
    htmldata += "<h4>Results:</h4>";
    _.each(htmlcontent.res, function(n){
      htmldata += "<a href='"+n+"' target='_blank'>"+n+"</a><br>";
    });
    uriContent = "data:application/octet-stream," + encodeURIComponent(htmldata);
    fname = 'bw_links_'+task.tid+((onlyValid)? '_v':'')+'.html';
    apputils.saveAs(uriContent, fname);
  };

  var updateTableStateParams = function(){
    $scope.stateslist = GenStates.find({}, {sort: {"_id": 1}}).fetch();
  };

  $scope.curState = new apputils.cCurrentState();

  $scope.remState = function(sid:string){
    if(sid){
      GenStates.remove(sid);
      updateTableStateParams();
      toaster.pop('warning', 'Removed ' + sid);
    }
  };

  $scope.chooseState = function(sid:string){
    $scope.enableImpSave = false;
    //we must get the state for this sid
    $scope.subscribe("genstates",()=>{return [sid]}, {
      onReady: function(sub){
         var myframe:iGenStates = GenStates.findOne({_id: sid});
         if(!myframe) return $scope.$apply(function(){toaster.pop('warn', 'Invalid State ID')});
         $scope.curState.clear();
         $scope.curState.copy(myframe);
         $scope.showMove(0);
      },
      onStop: subErr
    })
  };

  $scope.showMove = function(i:number){
    $('#imgpreview').empty();
    var scid:string = $scope.curState.block_states[i].screencapid;
    $scope.subscribe('screencaps', ()=>{return [scid]}, {
      onReady: function (sub) {
        var retid:string = navImgButtons('imgpreview', i);
        var screen:iScreenCaps = ScreenCaps.findOne({_id: scid});
        showImage(screen.data, 'Move #: ' + i, retid);
      },
      onStop: subErr
    });
  };

  var navImgButtons = function(id:string, i:number):string{
    var lenID:number = $('div').length;
    var eleDivID:string = 'rowdiv' + lenID; // Unique ID
    var retId:string = id+lenID;
    var htmlout = '';
    if(i < $scope.curState.block_states.length-1)
      htmlout += '<button onclick="angular.element(this).scope().showMove('+(i+1)+')" class="btn btn-xs btn-info pull-right" style="margin-left: 6px"> &gt; </button>';
    if(i > 0)
      htmlout += '<button onclick="angular.element(this).scope().showMove('+(i-1)+')" class="btn btn-xs btn-info pull-right"> &lt; </button>';
    htmlout += '<div id="'+retId+'"></div>';
    var attachTo = '#'+id;
    $('<div>').attr({
      id: eleDivID
    }).addClass('col-sm-12')
      .html(htmlout).css({"border-bottom": '1px solid #e4eaec'}).appendTo(attachTo);
    return retId;
  };

  var showImage = function(b64i:string, text:string, attachID:string){
    if(!attachID) return console.warn('showImage missing attachID');
    var b64img = LZString.decompressFromUTF16(b64i);

    var eleDivID:string = 'div' + $('div').length; // Unique ID
    var eleImgID:string = 'img' + $('img').length; // Unique ID
    //var eleLabelID:string = 'h4' + $('h4').length; // Unique ID
    var htmlout:string = '';
    if(text) htmlout += '<b>'+text+'</b><br>';
    htmlout += '<img id="'+eleImgID+'" style="width:'+canvas.width*4/5+'px;height:'+canvas.height*4/5+'px"></img>';
    // + '<label id="'+eleLabelID+'" class="mb"> '+id+'</label>';
    $('<div>').attr({
      id: eleDivID
    }).addClass('col-sm-4')
      .html(htmlout).css({}).appendTo('#'+attachID);

    var img:HTMLImageElement = <HTMLImageElement>document.getElementById(eleImgID); // Use the created element
    img.src = b64img;
  };

  $scope.taskGen = function(tasktype:string, movedir:string, bundle:number, asncnt:number, antcnt:number){
    var statelist:string[] = apputils.mdbArray(GenStates, {}, {
      sort: {"_id": 1}}, "_id");
    if(statelist.length){
      var jobdata:miGenJobsMgr.iGenJobsMgr = {
        stateid: $scope.curState._id,
        tasktype: tasktype,
        bundle: bundle,
        asncnt: asncnt,
        antcnt: antcnt,
        creator: $rootScope.currentUser._id,
        created: (new Date).getTime(),
        public: true,
        islist: true,
        list: null
      };
      if($scope.curState.type) jobdata.statetype = $scope.curState.type; //check if this state is partial/full or none
      
      var availlist:number[][] = [];
      var statelen:number = $scope.curState.block_states.length;
      //generate action jobs from states
      var doneAvailList = _.after(statelen, function(){
        var bundleidlist:string[] = [];
        var bundcnt:number = Math.ceil(availlist.length/jobdata.bundle);
        var doneBundles = _.after(bundcnt, function(){
          jobdata.list = bundleidlist;
          GenJobsMgr.insert(jobdata, function(err:Error, id:string) {
            if (!err) {
              updateJobMgr();
              toaster.pop('info', 'Jobs Created', id);
            }
            else toaster.pop('error', 'Job Create Error', err.message);
          })
        });

        function saveBundle(){
          var mybundledata:miGenJobsMgr.iGenJobsMgr = {
            stateid: $scope.curState._id,
            islist: false,
            tasktype: jobdata.tasktype,
            asncnt: jobdata.asncnt,
            antcnt: jobdata.antcnt,
            creator: $rootScope.currentUser._id,
            created: (new Date).getTime(),
            public: jobdata.public,
            idxlist: abundle
          };
          if(jobdata.statetype) mybundledata.statetype = $scope.curState.type;
          GenJobsMgr.insert(mybundledata, function(err:Error, id:string) {
            if (!err) {
              bundleidlist.push(id);
              doneBundles();
              toaster.pop('info', 'Bundle Created', id);
            }
            else toaster.pop('error', 'Bundle Data', err.message);
          });
          abundle = [];
        }
        var abundle:number[][] = [];
        for(var i:number = 0; i < availlist.length; i++){
          if(!(i % jobdata.bundle) && i) saveBundle();
          abundle.push(availlist[i]);
        }
        if(abundle.length) saveBundle(); //save the dangling bundles
      });

      //decide on normal or reverse action
      if(tasktype == 'action' && movedir == 'reverse'){
        for(var i:number = statelen-1; i > -1; i--){
          if(i > 0) availlist.push([i, i-1]); //because we use i & i+1 states in actions
          doneAvailList();
        }
      }
      else{
        for(var i:number = 0; i < statelen; i++){
          if(tasktype == 'action'){
            if(i < statelen-1) availlist.push([i, i+1]); //because we use i & i+1 states in actions
          }
          else availlist.push([i]);
          doneAvailList();
        }
      }
    }
  };
  
  var updateJobMgr = function(){
    $scope.jobmgrlist = GenJobsMgr.find({islist: true}, {sort: {"_id": 1}}).fetch();
  };
  
  $scope.selectJob = function(jid:string){
    var job:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: jid});
    $scope.jobid = jid;
    $scope.jobinfo = [];
    job.list.forEach(function(tid){
      var task:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: tid});
      $scope.jobinfo.push(task);
    });
  };
  
  $scope.remJob = function(jid:string){
    $scope.jobid = null;
    $scope.jobinfo = null; //null out job in case its the one deleted
    var deljob:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: jid});
    deljob.list.forEach(function(j){
      var deltask:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: j});
      if(deltask && deltask.hitlist)
        deltask.hitlist.forEach(function(h){
          GenJobsMgr.remove(h);
        });
      GenJobsMgr.remove(j);
    });
    GenJobsMgr.remove(jid);
    updateJobMgr();
    $scope.goodHITsData = false;
  };

  $scope.remHIT = function(tid:string, hid: string){
    $scope.jobid = null;
    $scope.jobinfo = null; //null out job in case its the one deleted
    var deltask:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: tid});
    if(deltask && deltask.hitlist) {
      GenJobsMgr.remove(hid);
      GenJobsMgr.update({_id: tid}, {$pull: {hitlist: hid}});
      var idx = _.findIndex($scope.allHITs.active, (a:iSortHITs)=>{return (tid === a.tid)});
      if(idx > -1) $scope.allHITs.active.splice(idx, 1);
      $scope.goodHITsData = false;
    }
  };

  $scope.addAsn = function(hid: string){
    var hit:miGenJobsMgr.iGenJobsHIT = GenJobsMgr.findOne({_id: hid});
    if(hit) {
      GenJobsMgr.update({_id: hid}, {$set: {"hitcontent.MaxAssignments": hit.hitcontent.MaxAssignments+1}});
      $scope.goodHITsData = false;
      toaster.pop('info', 'HIT '+hid+' assignment added.  Refresh to view updates.  This does not update MTurk job.');
    }
  };

  $scope.subAsn = function(hid: string){
    var hit:miGenJobsMgr.iGenJobsHIT = GenJobsMgr.findOne({_id: hid});
    if(hit) {
      GenJobsMgr.update({_id: hid}, {$set: {"hitcontent.MaxAssignments": hit.hitcontent.MaxAssignments-1}});
      $scope.goodHITsData = false;
      toaster.pop('info', 'HIT '+hid+' assignment decremented.  Refresh to view updates.  This does not update MTurk job.');
    }
  };

  $scope.createHIT = function(jid:string, tid:string){
    var params:iTurkCreateParam = {jid: jid, tid: tid, islive: $scope.opt.isLive, useQual: $scope.opt.useQual};
    Meteor.call('mturkCreateHIT', params, function(err, ret){
      if(err) return $scope.$apply(function(){toaster.pop('error', err)});
      if(ret.error) return $scope.$apply(function(){toaster.pop('error', ret.error)});
      //create the HITId system
      var res = ret.result;
      var hitdata:miGenJobsMgr.iGenJobsHIT = {
        '_id': 'H_'+res.hit[0].HITId,
        HITId: res.hit[0].HITId,
        HITTypeId: res.hit[0].HITTypeId,
        hitcontent: res.hitcontent,
        tid: tid,
        jid: jid,
        islive: $scope.opt.isLive,
        created: (new Date()).getTime()
      };
      $scope.$apply(function(){toaster.pop('info', 'HIT created: '+ hitdata._id)});
      //cannot use save with custom _id
      GenJobsMgr.insert(hitdata, function(err, hid){
        if(err) return $scope.$apply(function(){toaster.pop('error', err)});
        GenJobsMgr.update({_id: tid}, {$addToSet: {hitlist: hid}});
        $scope.selectJob($scope.jobid);
        $scope.goodHITsData = false;
      });
    });
  };

  $scope.blockTurker = function(tuid:string, reason: string){
    var params:iBlockTurker = {WorkerId: tuid, Reason: reason};
    Meteor.call('mturkBlockTurker', params, function(err, ret){
      if(err) return $scope.$apply(function(){toaster.pop('error', JSON.stringify(err, null, 2))});
      if(ret.error) return $scope.$apply(function(){toaster.pop('error', JSON.stringify(ret.error, null, 2))});
      $scope.$apply(function(){toaster.pop('info', JSON.stringify(ret.result, null, 2))});
    });
  };

  $scope.getReviewHITs = function(s:string){
    var params:iReviewableHITs = {Status: s, PageSize: 20, PageNumber: 1};
    Meteor.call('mturkReviewHITs', params, function(err, ret){
      if(err) return $scope.$apply(function(){toaster.pop('error', JSON.stringify(err, null, 2))});
      if(ret.error) return $scope.$apply(function(){toaster.pop('error', JSON.stringify(ret.error, null, 2))});
      $scope.$apply(function(){toaster.pop('info', 'HITs', JSON.stringify(ret.result, null, 2))});
    });
  };

  interface iJTHInfo{
    sid:string,
    jid:string,
    tid:string,
    hid:string,
    url:string
  }
  $scope.getURLHITs = function(jidstr){
    var jids:string[] = jidstr.trim().split(/[ ,]/);
    $scope.subscribe('genjobsmgr', ()=>{return[{type:'item', keys: jids}]}, {
      onReady: (sub)=>{
        var myjobs:miGenJobsMgr.iGenJobsMgr[] = GenJobsMgr.find({_id: {$in: jids}}).fetch();
        if(myjobs.length){
          var jtids:{tid:string, jid:string}[] = [];
          var tids:string[] = [];
          _.each(myjobs, function(job:miGenJobsMgr.iGenJobsMgr){
            _.each(job.list, function(tid) {
              jtids.push({tid: tid, jid:job._id});
              tids.push(tid);
            });
          });
          $scope.subscribe('genjobsmgr', ()=>{return[{type:'item', keys: tids}]}, {
            onReady: (sub)=>{
              var turkreqlink = 'https://requester.mturk.com/mturk/manageHIT?viewableEditPane=&HITId=';
              var hitlist:iJTHInfo[] = [];
              _.each(jtids, function(jtid){
                var mytask:miGenJobsMgr.iGenJobsMgr = GenJobsMgr.findOne({_id: jtid.tid});
                _.each(mytask.hitlist, function(h){
                  var hid = h.replace(/H_/,'');
                  hitlist.push({jid: jtid.jid, tid: jtid.tid, hid: hid, sid: mytask.stateid, url: turkreqlink+hid});
                });
              });
              var dialog = ngDialog.open({
                template: 'didTurkURLs',
                data: hitlist,
                className: 'ngdialog-theme-default width60perc',
                controller: ['$scope', function($scope){
                }]
              });
              dialog.closePromise.then(function(data){
                console.log('ngDialog closed', data);
                if(data.value){
                }
              });
            }
            ,onStop: subErr
          });
        }
        else toaster.pop('warning','Job ID not found: '+JSON.stringify(jids));
      }
      ,onStop: subErr
    });
  };

  $scope.stateGo = apputils.stateGo($state);
}]);
