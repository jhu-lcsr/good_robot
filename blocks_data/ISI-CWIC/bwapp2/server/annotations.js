/**========================================================
 * Module: annotations.js
 * Created by wjwong on 9/3/15.
 =========================================================*/

Annotations.allow({
 insert: function(userId, job){
  return userId && job.owner === userId;
 },
 update: function(userId, job, fields, modifier){
  return userId && (job.owner === userId || job.agent === userId);
 },
 remove: function(userId, job){
  return userId && job.owner === userId;
 },
 fetch: ['owner', 'agent']
});

Meteor.publish('annotations', function(){
 return Annotations.find({
  $and: [
   {owner: this.userId},
   {owner: {$exists: true}}
  ]
 });
});
