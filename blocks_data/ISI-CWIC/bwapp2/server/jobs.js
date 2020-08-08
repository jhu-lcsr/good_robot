/**========================================================
 * Module: jobs.js
 * Created by wjwong on 8/11/15.
 =========================================================*/

Jobs.allow({
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

Meteor.publish('jobs', function(){
  return Jobs.find({
    $and: [
     {owner: this.userId},
     {owner: {$exists: true}}
    ]
  });
});

Meteor.publish('agentjobs', function(){
  return Jobs.find({
    $and: [
      {agent: this.userId},
      {agent: {$exists: true}}
    ]
  });
});
