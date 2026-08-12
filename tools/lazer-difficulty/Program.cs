using System.Reflection;
using System.Runtime.Loader;
using System.Text.Json;

internal static class Program
{
    private static string lazerDirectory = string.Empty;
    private static Assembly gameAssembly = null!;
    private static Assembly osuRulesetAssembly = null!;
    private static string gameVersion = "unknown";

    private static int Main(string[] args)
    {
        try
        {
            if (args.Length is < 1 or > 2)
                throw new ArgumentException(
                    "Usage: OsuMapper.LazerDifficulty <lazer-directory> [beatmap.osu]");

            lazerDirectory = Path.GetFullPath(args[0]);
            if (!Directory.Exists(lazerDirectory))
                throw new DirectoryNotFoundException($"osu!lazer directory does not exist: {lazerDirectory}");

            AssemblyLoadContext.Default.Resolving += resolveLazerAssembly;
            gameAssembly = loadAssembly("osu.Game.dll");
            osuRulesetAssembly = loadAssembly("osu.Game.Rulesets.Osu.dll");
            registerDecoderDependencies(gameAssembly);
            Version? assemblyVersion = gameAssembly.GetName().Version;
            gameVersion = gameAssembly
                .GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                ?.InformationalVersion
                ?? assemblyVersion?.ToString()
                ?? "unknown";

            if (args.Length == 2)
            {
                Console.WriteLine(calculateJson(args[1]));
                return 0;
            }

            string? line;
            while ((line = Console.ReadLine()) is not null)
            {
                try
                {
                    Console.WriteLine(calculateJson(line));
                }
                catch (Exception exception)
                {
                    Console.WriteLine(JsonSerializer.Serialize(new { error = unwrap(exception).Message }));
                }
                Console.Out.Flush();
            }
            return 0;
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine(unwrap(exception).Message);
            return 2;
        }
    }

    private static string calculateJson(string path)
    {
        string beatmapPath = Path.GetFullPath(path);
        if (!File.Exists(beatmapPath))
            throw new FileNotFoundException("Beatmap does not exist", beatmapPath);
        Type workingBeatmapType = requireType(gameAssembly, "osu.Game.Beatmaps.FlatWorkingBeatmap");
        object workingBeatmap = createWorkingBeatmap(workingBeatmapType, beatmapPath);
        Type rulesetType = requireType(osuRulesetAssembly, "osu.Game.Rulesets.Osu.OsuRuleset");
        object ruleset = Activator.CreateInstance(rulesetType)
            ?? throw new InvalidOperationException("Could not create the osu!standard ruleset.");
        object calculator = createCalculator(ruleset, workingBeatmap);
        object attributes = calculate(calculator);
        double stars = readStarRating(attributes);
        return JsonSerializer.Serialize(new
        {
            stars,
            calculator = "osu!lazer-installed",
            game_version = gameVersion,
        });
    }

    private static Assembly? resolveLazerAssembly(AssemblyLoadContext context, AssemblyName name)
    {
        string candidate = Path.Combine(lazerDirectory, $"{name.Name}.dll");
        return File.Exists(candidate) ? context.LoadFromAssemblyPath(candidate) : null;
    }

    private static Assembly loadAssembly(string filename)
    {
        string path = Path.Combine(lazerDirectory, filename);
        if (!File.Exists(path))
            throw new FileNotFoundException($"Required osu!lazer assembly is missing: {path}", path);
        return AssemblyLoadContext.Default.LoadFromAssemblyPath(path);
    }

    private static Type requireType(Assembly assembly, string name) =>
        assembly.GetType(name, throwOnError: true)
        ?? throw new TypeLoadException($"osu!lazer type is missing: {name}");

    private static void registerDecoderDependencies(Assembly gameAssembly)
    {
        Type decoderType = requireType(gameAssembly, "osu.Game.Beatmaps.Formats.Decoder");
        Type storeType = requireType(gameAssembly, "osu.Game.Rulesets.AssemblyRulesetStore");
        object store = createWithDefaults(storeType, lazerDirectory);
        MethodInfo method = decoderType.GetMethods(BindingFlags.Public | BindingFlags.Static)
            .Single(candidate =>
                candidate.Name == "RegisterDependencies"
                && candidate.GetParameters().Length == 1
                && candidate.GetParameters()[0].ParameterType.IsInstanceOfType(store));
        method.Invoke(null, new[] { store });
    }

    private static object createWorkingBeatmap(Type type, string path)
    {
        ConstructorInfo constructor = type.GetConstructors()
            .FirstOrDefault(candidate =>
            {
                ParameterInfo[] parameters = candidate.GetParameters();
                return parameters.Length >= 1 && parameters[0].ParameterType == typeof(string);
            }) ?? throw new MissingMethodException(type.FullName, ".ctor(string, ...)");
        ParameterInfo[] parameters = constructor.GetParameters();
        object?[] values = parameters
            .Select((parameter, index) => index == 0 ? path : optionalValue(parameter))
            .ToArray();
        return constructor.Invoke(values);
    }

    private static object createWithDefaults(Type type, string? firstString = null)
    {
        ConstructorInfo constructor = type.GetConstructors()
            .OrderBy(candidate => candidate.GetParameters().Length)
            .FirstOrDefault()
            ?? throw new MissingMethodException(type.FullName, ".ctor");
        object?[] values = constructor.GetParameters()
            .Select((parameter, index) =>
                index == 0 && parameter.ParameterType == typeof(string) && firstString is not null
                    ? firstString
                    : optionalValue(parameter))
            .ToArray();
        try
        {
            return constructor.Invoke(values);
        }
        catch (Exception exception)
        {
            string signature = string.Join(", ", constructor.GetParameters().Select(parameter =>
                $"{parameter.ParameterType.FullName} {parameter.Name} default={parameter.DefaultValue ?? "null"}"));
            throw new InvalidOperationException(
                $"Could not construct {type.FullName}({signature}): {unwrap(exception).Message}",
                exception);
        }
    }

    private static object createCalculator(object ruleset, object workingBeatmap)
    {
        MethodInfo method = ruleset.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Single(candidate =>
                candidate.Name == "CreateDifficultyCalculator"
                && candidate.GetParameters().Length == 1
                && candidate.GetParameters()[0].ParameterType.IsInstanceOfType(workingBeatmap));
        return method.Invoke(ruleset, new[] { workingBeatmap })
            ?? throw new InvalidOperationException("osu!lazer did not create a difficulty calculator.");
    }

    private static object calculate(object calculator)
    {
        MethodInfo method = calculator.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(candidate => candidate.Name == "Calculate")
            .OrderBy(candidate => candidate.GetParameters().Length)
            .First(candidate => candidate.GetParameters().All(parameter => parameter.IsOptional));
        object?[] values = method.GetParameters().Select(optionalValue).ToArray();
        return method.Invoke(calculator, values)
            ?? throw new InvalidOperationException("osu!lazer returned no difficulty attributes.");
    }

    private static double readStarRating(object attributes)
    {
        PropertyInfo property = attributes.GetType().GetProperty("StarRating")
            ?? throw new MissingMemberException(attributes.GetType().FullName, "StarRating");
        object value = property.GetValue(attributes)
            ?? throw new InvalidOperationException("osu!lazer returned a null star rating.");
        return Convert.ToDouble(value);
    }

    private static object? optionalValue(ParameterInfo parameter)
    {
        if (parameter.HasDefaultValue && parameter.DefaultValue is not DBNull)
            return parameter.DefaultValue;
        return parameter.ParameterType.IsValueType
            ? Activator.CreateInstance(parameter.ParameterType)
            : null;
    }

    private static Exception unwrap(Exception exception)
    {
        while (exception is TargetInvocationException { InnerException: not null } invocation)
            exception = invocation.InnerException!;
        return exception;
    }
}
