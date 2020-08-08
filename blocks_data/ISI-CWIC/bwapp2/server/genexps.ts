/**========================================================
 * Module: genexps
 * Created by wjwong on 12/16/15.
 =========================================================*/
/// <reference path="../model/genexpsdb.ts" />
/// <reference path="./typings/meteor/meteor.d.ts" />
/// <reference path="./typings/lodash/lodash.d.ts" />

var validKeys:string[] = ['_id', 'public', 'block_meta', 'block_state', 'created', 'name', 'creator', 'utterance'];

GenExps.allow({
  insert: function(userId, data){
    var fcheck:string[] = _.difference(_.keys(data), validKeys);
    if(fcheck.length) throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
    return userId;
  },
  update: function(userId, data, fields, modifier){
    var fcheck:string[] = _.difference(_.keys(data), validKeys);
    if(fcheck.length) throw new Match['Error']("illegal fields:" + JSON.stringify(fcheck));
    return userId;
  },
  remove: function(userId, data){
    return userId;
  }
  ,fetch: ['_id']
});

Meteor.publish('genexps', function(id){
  if(id){
    return GenExps.find({
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
  else return GenExps.find({
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
