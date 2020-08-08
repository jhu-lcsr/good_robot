/**
 * Created by wjwong on 7/26/15.
 */
Meteor.startup(function () {
  var userlist = [
    //{username: 'testuser@company.com', roles: ['admin'], pwd: 'password'},
  ];

  _.each(userlist, function(usr){
    if(!Meteor.users.findOne({username: usr.username})){
      try{
        var userid = Accounts.createUser({
          username: usr.username,
          email: usr.username,
          password: usr.pwd,
          profile: {roles: usr.roles}
        });
        console.warn(usr.username, userid)
      }
      catch(err){
        console.warn(err)
      }
    }
  })
});