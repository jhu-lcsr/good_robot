/**========================================================
 * Module: screencaps.js
 * Created by wjwong on 10/3/15.
 =========================================================*/

ScreenCaps.allow({
  insert: function(userId, job){
    return userId;
  },
  update: function(userId, job, fields, modifier){
    return userId;
  },
  remove: function(userId, job){
    return userId;
  },
  fetch: ['_id']
});

Meteor.publish('screencaps', function(id){
  if(id){
    if(id.constructor === Array){
      return ScreenCaps.find({
        $and: [
          {
            $and: [
              {'public': true},
              {'public': {$exists: true}}
            ]
          }
          ,{'_id': { $in: id}}
        ]});
    }
    else return ScreenCaps.find({
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
  else
    return ScreenCaps.find({
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
      }
      ,{fields: {'_id': 1, 'public': 1}}
    );
});
