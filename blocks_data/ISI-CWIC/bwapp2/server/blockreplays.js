/**========================================================
 * Module: blockreplays
 * Created by wjwong on 8/7/15.
 =========================================================*/
BlockReplays.allow({
  insert: function(userId, replay){
    var fcheck = _.without(_.keys(replay), 'name', 'owner', 'creator', 'created', 'start', 'end', 'public','data');
    if(fcheck.length) throw new Match.Error("illegal fields:"+JSON.stringify(fcheck));
    return userId && replay.owner === userId;
  },
  update: function(userId, replay, fields, modifier){
    var fcheck = _.without(_.keys(replay), '_id', 'name', 'owner', 'creator', 'created', 'start', 'end', 'public','data');
    if(fcheck.length) throw new Match.Error("illegal fields:"+JSON.stringify(fcheck));
    return userId && replay.owner === userId;
  },
  remove: function(userId, replay){
    return userId && replay.owner === userId;
  },
  fetch: ['owner']
});

Meteor.publish('blockreplays', function(){
  return BlockReplays.find({
    $or: [
      {$and: [
        {'public': true},
        {'public': {$exists: true}}
      ]},
      {$and: [
        {owner: this.userId},
        {owner: {$exists: true}}
      ]}
    ]
  });
});
