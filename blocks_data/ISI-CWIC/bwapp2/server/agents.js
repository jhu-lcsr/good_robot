/**========================================================
 * Module: agents.js
 * Created by wjwong on 8/11/15.
 =========================================================*/

Meteor.publish("agents", function(){
  return Meteor.users.find(
    {$and: [
      {'profile.roles': 'agent'},
      {'profile.roles': {$exists: true}}
    ]},
    {
      fields: {emails: 1, username: 1, profile: 1},
      sort: {username: 1}
    });
});