/**========================================================
 * Module: gencmdjobs
 * Created by wjwong on 2/5/16.
 =========================================================*/
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="../model/gencmdjobsdb.ts" />
/// <reference path="./util.ts" />

GenCmdJobs.allow({
  insert: function(userId, job):boolean{
    //console.warn('insert');
    if(isRole(Meteor.user(), 'guest')) return false;
    return (userId)?true:false; // && job.owner === userId;
  },
  update: function(userId, job, fields, modifier):boolean{
    if(isRole(Meteor.user(), 'guest')){
      var idx = '$set';
      var delkeys:string[] = [];
      //only allow pass through of updates to notes and submitted
      if(modifier['$set']){
        var keys:string[] = Object.keys(modifier['$set']);
        _.each(keys, function(k){
          if(!(k.match(/notes/g) || k.match(/timed/g))) delkeys.push(k);
        });
        //keys = _.difference(keys, ['notes', 'submitted', 'timed']); //only allow notes and submitted when not logged in
      }
      else{
        if(modifier['$addToSet']){
          idx = '$addToSet';
          var keys:string[] = Object.keys(modifier['$addToSet']);
          var delkeys:string[] = [];
          _.each(keys, function(k){
            if(!k.match(/submitted/g)) delkeys.push(k);
          });
        }
        else return false;
      }

      if(delkeys.length) console.warn('GenCmdJobs '+idx+' del: ');
      for(var i = 0; i < delkeys.length; i++){
        console.warn(delkeys[i]);
        delete modifier[idx][delkeys[i]];
      }
    }
    return (userId)?true:false;
  },
  remove: function(userId, job):boolean{
    if(isRole(Meteor.user(), 'guest')) return false;
    return (userId)?true:false; // && job.owner === userId;
  }
  //,fetch: ['owner']
});

Meteor.publish('gencmdjobs', function(params?: {type: string}):Mongo.Cursor<miGenCmdJobs.iGenCmdJobs>{
  if(params){
    switch(params.type){
      case 'submitted':
        return GenCmdJobs.find(
          {$and: [{HITId: {$exists: true}}, {submitted: {$exists: true}}]}
          , {fields: {tid: 1, submitted: 1}, sort: {'submitted.time': -1}}
        );
      case 'list':
        return GenCmdJobs.find(
          {}
          , {sort: {'created': -1}, skip: params['pageSize'] * params['pageCur'], limit: params['pageSize']}
        );
        /*
         return GenCmdJobs.find(
         {HITId: {$exists: true}}
         , {fields: {tid: 1, jid: 1, 'submitted.name': 1, 'submitted.valid': 1, 'submitted.time': 1, 'hitcontent.MaxAssignments': 1, 'hitcontent.Reward': 1, 'created': 1, 'islive': 1}, sort: {'created': -1}, skip: params['pageSize'] * params['pageCur'], limit: params['pageSize']}
         );*/
      case 'item':
        if(Array.isArray(params['keys'])) {
          return GenCmdJobs.find(
            {_id: {$in: params['keys']}}
          );
        }
        else this.error(555, 'missing keys array');
        break;
      default:
        console.warn('gencmdjobs pub missing type[submitted, list, item]');
    }
  }
  else return GenCmdJobs.find({
      /*$and: [
       {owner: this.userId},
       {owner: {$exists: true}}
       ]*/
    }
    //, {fields: {'_id': 1, 'islist': 1}}
  );
});


