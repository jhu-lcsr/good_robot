/**========================================================
 * Module: gencmds
 * Created by wjwong on 1/26/16.
 =========================================================*/
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />
/// <reference path="../model/gencmdsdb.ts" />
/// <reference path="./util.ts" />

var validKeys:string[] = ['_id', 'public', 'block_meta', 'block_state', 'created', 'name', 'creator', 'type'];

GenCmds.allow({
  insert: function(userId, data){
    if(isRole(Meteor.user(), 'guest')) return false;
    var fcheck = _.difference(_.keys(data), validKeys);
    if(fcheck.length) throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
    return userId;
  },
  update: function(userId, data, fields, modifier){
    if(isRole(Meteor.user(), 'guest')) return false;
    var fcheck = _.difference(_.keys(data), validKeys);
    if(fcheck.length) throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
    return userId;
  },
  remove: function(userId, data){
    if(isRole(Meteor.user(), 'guest')) return false;
    return userId;
  }
  ,fetch: ['_id']
});

Meteor.publish('gencmds', function(id){
  if(id){
    return GenCmds.find({
      $and: [
        {
          $and: [
            {'public': true},
            {'public': {$exists: true}}
          ]
        }
        ,{'_id': id}
      ]});
  }
  else return GenCmds.find({
      $or: [
        {
          $and: [
            {'public': true},
            {'public': {$exists: true}}
          ]
        }
        /*,
         {$and: [
         {owner: this.userId},
         {owner: {$exists: true}}
         ]}*/
      ]
    },
    {fields: {'_id': 1, 'stateitr': 1, 'name': 1, 'created': 1}}
  );
});
